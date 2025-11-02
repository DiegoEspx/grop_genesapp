from __future__ import annotations
import re, logging
from typing import List, Dict, Tuple
from llm_client import build_llm_from_env
from bm25_store import (
    upsert_document as bm25_upsert,
    query_context as bm25_query,
    format_apa6_list as bm25_apa,
    delete_document as bm25_delete,
    doc_stats as bm25_stats,
    list_docs as bm25_list,
)

# --- Configuración base ---
log = logging.getLogger("uvicorn.error")
_LLM_CLIENT, _LLM_BACKEND, _LLM_MODEL = build_llm_from_env()

def _chat(messages, model, **opts):
    return _LLM_CLIENT.chat(messages, **opts)


# --- Palabras clave y temas reconocidos ---
_HEALTH_KEYWORDS = {
    "salud", "síntoma", "sintoma", "diagnóstico", "diagnostico",
    "enfermedad", "síndrome", "sindrome", "genética", "genetica",
    "tratamiento", "prevención", "prevencion", "terapia", "cuidado",
    "manifestación", "manifestacion", "característica", "caracteristica"
}

_DISEASE_ALIASES = {
    r"\bdown\b": "down",
    r"\bwilliams\b": "williams",
    r"\bmps\b": "mps",
    r"\bmucopolisacaridos(?:is|es)?\b": "mps",
    r"\bmucopolisacaridosis\b": "mps",
    r"\bhurler\b": "mps",
    r"\bhunter\b": "mps",
    r"\bsanfilippo\b": "mps",
    r"\bmorquio\b": "mps",
}

# --- Gemi: personalidad del asistente ---
_GEMI_NAME = "Gemi"
_GREET_WORDS = {
    "hola", "buenas", "buenos días", "buenas tardes", "buenas noches",
    "hey", "hi", "hello", "qué tal", "que tal"
}
_SMALLTALK_WORDS = {
    "cómo estás", "como estas", "que haces", "qué haces",
    "cómo va", "como va", "gracias", "ok", "vale"
}

def _is_greeting(t: str) -> bool:
    t = t.lower().strip()
    return any(w in t for w in _GREET_WORDS)

def _is_smalltalk(t: str) -> bool:
    t = t.lower().strip()
    return _is_greeting(t) or any(w in t for w in _SMALLTALK_WORDS)


# --- Funciones auxiliares ---
def _is_health_related(text: str) -> bool:
    t = (text or "").lower()
    if any(kw in t for kw in _HEALTH_KEYWORDS):
        return True
    return _extract_topic(t) is not None

def _extract_topic(text: str) -> str | None:
    for pat, topic in _DISEASE_ALIASES.items():
        if re.search(pat, text.lower()):
            return topic
    return None

def _tidy_output(text: str, max_sentences: int = 15) -> str:
    """Limpia salida, normaliza viñetas y evita frases pegadas/cortadas."""
    if not text:
        return text
    # * -> viñetas
    text = re.sub(r"(?m)^\s*\*\s+", "• ", text)
    # separa frases que quedaron pegadas al pasar de línea
    text = re.sub(r"([a-záéíóúñ])\s+([A-ZÁÉÍÓÚÑ])", r"\1. \2", text)
    # espacios repetidos
    text = re.sub(r"\s{2,}", " ", text).strip()
    # limitar longitud por oraciones
    sents = re.split(r"(?<=[\.\?\!])\s+", text)
    text = " ".join(sents[:max_sentences]).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


# --- Construcción de filtros para el RAG (BM25) ---
def _compose_where(topic: str | None = None,
                   lang: str | None = None,
                   min_year: int | None = None,
                   types: list[str] | None = None) -> dict | None:
    parts = []
    if topic:
        parts.append({"topic": {"$eq": topic}})
    if lang:
        parts.append({"lang": {"$eq": lang}})
    if min_year is not None:
        parts.append({"year": {"$gte": min_year}})
    if types:
        parts.append({"type": {"$in": types}})
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


# --- Generación de respuesta ---
def generate_answer(user_msg: str,
                    topic: str | None = None,
                    min_year: int | None = None,
                    types: list[str] | None = None,
                    lang: str | None = None):
    t = (user_msg or "").strip()

    # Small-talk / saludo
    if _is_smalltalk(t):
        return (
            f"¡Hola! Soy {_GEMI_NAME}, tu asistente especializado en enfermedades raras. "
            "Puedo ayudarte con información sobre síndrome de Down, Williams y mucopolisacaridosis (MPS). "
            "¿Qué te gustaría saber?",
            [], []
        )

    inferred = _extract_topic(t)

    # Si no hay señales médicas NI topic dado
    if not _is_health_related(t) and not (topic or inferred):
        return (
            "Me especializo en enfermedades raras y educación en salud genética. "
            "¿Podrías indicarme sobre qué condición te gustaría información? "
            "Puedo ayudarte con Down, Williams o MPS.",
            [], []
        )

    effective_topic = topic or inferred
    where = _compose_where(topic=effective_topic, lang=lang, min_year=min_year, types=types)

    # Recuperar contexto con BM25 filtrado
    rag_text, metas = bm25_query(t, k=5, where=where)

    # NUEVO: Sistema de prompts mejorado que permite conocimiento general + documentos
    SYSTEM_PROMPT = f"""Eres Gemi, un asistente médico educativo especializado en enfermedades raras, con amplio conocimiento en genética médica, pediatría y síndromes poco frecuentes.

INSTRUCCIONES CLAVE:
1. Combina tu conocimiento médico general con la información de los documentos proporcionados
2. Los documentos son una BASE para asegurar precisión, NO tu única fuente
3. Puedes explicar conceptos, fisiopatología, epidemiología y manejo general usando tu conocimiento médico
4. Cuando algo esté en los documentos, menciónalo como respaldo ("según la literatura...")
5. Responde de forma natural, clara y educativa, como un médico explicando a un estudiante
6. Usa 6-12 oraciones para respuestas completas
7. Si explicas manifestaciones o tratamientos, usa viñetas con '• '
8. NUNCA des diagnósticos personales ni dosis específicas
9. Siempre responde en español

TEMA ACTUAL: {effective_topic or "enfermedades raras en general"}

Tu objetivo es EDUCAR de forma completa, no solo repetir documentos."""

    # Si hay documentos, los usamos como contexto de respaldo
    if rag_text:
        context_section = f"\n\nDOCUMENTOS DE REFERENCIA (úsalos para validar y complementar tu conocimiento):\n{rag_text}"
    else:
        context_section = "\n\nNOTA: No hay documentos específicos cargados para este tema, pero puedes usar tu conocimiento médico general sobre enfermedades raras."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + context_section},
        {"role": "user", "content": t},
    ]

    # Aumentamos tokens para respuestas más completas
    out = _chat(messages, _LLM_MODEL, temperature=0.35, num_predict=450, top_p=0.92)
    raw = (out.get("message") or {}).get("content", "").strip()
    reply = _tidy_output(raw, max_sentences=15)

    citations_apa = bm25_apa(metas) if metas else []
    
    # Nota de seguridad más natural
    if reply and not any(x in reply.lower() for x in ["consulta", "médico", "profesional"]):
        reply += "\n\nRecuerda: esta información es educativa. Para situaciones individuales, consulta con un profesional de la salud."

    return reply or "No pude generar una respuesta en este momento.", metas, citations_apa


# --- Reexportaciones (API pública del módulo) ---
upsert_document = bm25_upsert
list_docs = bm25_list
doc_stats = bm25_stats
delete_document = bm25_delete