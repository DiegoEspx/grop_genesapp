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
    "tratamiento", "prevención", "prevencion"
}

_DISEASE_ALIASES = {
    r"\bdown\b": "down",
    r"\bwilliams\b": "williams",
    r"\bmps\b": "mps",
    r"\bmucopolisacaridos(?:is|es)?\b": "mps",      # mucopolisacaridosis / mucopolisacaridoses
    r"\bmucopolisacaridosis\b": "mps",
    r"\bhurler\b": "mps",       # MPS I
    r"\bhunter\b": "mps",       # MPS II
    r"\bsanfilippo\b": "mps",   # MPS III
    r"\bmorquio\b": "mps",      # MPS IV
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

def _tidy_output(text: str, max_sentences: int = 10) -> str:
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
# El resto de imports, _chat, y funciones auxiliares permanecen igual...

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
            f"¡Hola! Soy {_GEMI_NAME}, tu asistente para dudas sobre enfermedades raras. "
            "¿Qué te gustaría saber? Puedo ayudarte con síndrome de Down, Williams o MPS.",
            [], []
        )

    inferred = _extract_topic(t)

    # Si no hay señales médicas NI topic dado (p.ej., repregunta sin contexto)
    if not _is_health_related(t) and not (topic or inferred):
        return (
            "Me centro en enfermedades raras y educación en salud. "
            "¿Podrías indicarme si te refieres a Down, Williams o MPS?",
            [], []
        )

    effective_topic = topic or inferred
    where = _compose_where(topic=effective_topic, lang=lang, min_year=min_year, types=types)

    # Recuperar contexto con BM25 filtrado
    rag_text, metas = bm25_query(t, k=7, where=where)

    # --------------------------------------------------------------------------------
    # NUEVA LÓGICA DE FALLBACK (SI NO HAY DOCUMENTOS RAG)
    # --------------------------------------------------------------------------------
    if not rag_text:
        # Usamos el LLM para una respuesta de conocimiento general ampliada
        SYSTEM_PROMPT_FALLBACK = (
            "Eres 'Gemi', un asistente educativo de salud, amable y experto. "
            "Estás respondiendo a una pregunta sobre una enfermedad rara (Síndrome de Down, Williams o MPS). "
            "**IMPORTANTE: No se han encontrado documentos locales (RAG).** Debes usar **solo tu conocimiento general** "
            "para responder la pregunta. Responde de forma clara y concisa (4-8 oraciones). **No digas que no tienes documentos, simplemente responde con seguridad**. "
            "Añade al final de tu respuesta la nota: '(Respuesta de conocimiento general, sin evidencia local)'. "
            "No hagas diagnósticos. Si listes, usa viñetas '• '."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FALLBACK},
            {"role": "user", "content": "Pregunta: " + t}
        ]

        # Usamos el LLM para una respuesta de conocimiento general dinámica y no repetitiva
        out = _chat(messages, _LLM_MODEL, temperature=0.35, num_predict=350, top_p=0.95)
        raw = (out.get("message") or {}).get("content", "").strip()
        reply = _tidy_output(raw)
        return reply or "No pude generar una respuesta de conocimiento general en este momento.", [], []
    
    # --------------------------------------------------------------------------------
    # FIN DE LA NUEVA LÓGICA DE FALLBACK
    # --------------------------------------------------------------------------------


    # --------------------------------------------------------------------------------
    # FLUJO CON RAG: Ahora con un SYSTEM_PROMPT más flexible
    # --------------------------------------------------------------------------------
    SYSTEM_PROMPT = (
        "Eres Gemi, un **asistente educativo de salud** experto, amigable y muy entusiasta, "
        "especializado en enfermedades raras como Síndrome de Down, Williams y MPS.\n"
        "Tu misión es ofrecer respuestas completas y de alta calidad. **Combina tu conocimiento general** para el contexto y la fluidez, "
        "y usa el **Contexto Recuperado (RAG)** como evidencia principal para datos específicos. No te limites solo a repetir los documentos.\n"
        "Responde SIEMPRE en español, con un tono cálido y proactivo. Usa viñetas '• ' cuando listes síntomas o características.\n"
        "No hagas diagnósticos, no indiques dosis ni tratamientos. **La respuesta debe ser concisa, idealmente 4-8 oraciones**.\n"
        "Marca con [1], [2], ... los fragmentos del contexto recuperado cuando los uses."
    )
    
    ctx_preview = "\n".join(rag_text.split("\n")[:7])  # 7 fragmentos/lineas como preview

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Contexto recuperado (fragmentos relevantes):\n{ctx_preview}"},
        {"role": "user", "content": t},
    ]

    out = _chat(messages, _LLM_MODEL, temperature=0.35, num_predict=350, top_p=0.95)
    raw = (out.get("message") or {}).get("content", "").strip()
    reply = _tidy_output(raw)

    citations_apa = bm25_apa(metas)
    if reply:
        # Añadimos la nota de seguridad al final de las respuestas con RAG
        if "No sustituye evaluación médica" not in reply:
            reply += " (Contenido educativo; no sustituye evaluación médica)."

    return reply or "No pude generar respuesta en este momento.", metas, citations_apa

# ... Reexportaciones (API pública del módulo) permanecen igual...


# --- Reexportaciones (API pública del módulo) ---
upsert_document = bm25_upsert
list_docs = bm25_list
doc_stats = bm25_stats
delete_document = bm25_delete
