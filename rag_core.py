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

def _generic_definition(topic: str | None) -> str | None:
    if topic == "down":
        return ("El síndrome de Down es un trastorno genético causado por la "
                "trisomía del cromosoma 21.")
    if topic == "williams":
        return ("El síndrome de Williams es una enfermedad genética por "
                "microdeleción en 7q11.23, con rasgos faciales típicos y "
                "frecuente afectación cardiovascular.")
    if topic == "mps":
        return ("Las mucopolisacaridosis (MPS) son trastornos lisosomales por "
                "déficit enzimático que impide degradar glucosaminoglucanos; "
                "cursan con afectación multisistémica progresiva.")
    return None


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

    # Recuperar contexto con BM25 filtrado (pedimos un poco más de contexto para dar más material)
    rag_text, metas = bm25_query(t, k=7, where=where)

    # Si no hay docs del tema → fallback educativo honesto
    if not rag_text:
        generic = _generic_definition(effective_topic)
        if generic:
            generic += " (Respuesta de conocimiento general; no tengo documentos locales para citar aún)."
            return (generic, [], [])
        return (
            "No tengo documentos locales para ese tema todavía. "
            "Puedes subir un PDF relacionado con /ingest y volver a preguntar.",
            [], []
        )

    # --- Nuevo SYSTEM_PROMPT: combinar conocimiento general + documentos locales ---
    SYSTEM_PROMPT = (
        "Eres 'Gemi', un asistente educativo en salud, claro y empático, especializado en enfermedades raras.\n"
        "Instrucciones estrictas para todas las respuestas:\n"
        "1) RESPONDE EN ESPAÑOL en un tono claro y humano (4-8 oraciones preferible para preguntas generales).\n"
        "2) Usa tu conocimiento general verificado para explicar conceptos relevantes, y USA LOS DOCUMENTOS LOCALES recuperados "
        "como fuentes de apoyo. No limites la respuesta a repetir textualmente los documentos.\n"
        "3) Si usas información específica que proviene de los documentos, indícalo con una cita entre corchetes numéricos "
        "ej.: [1]. Al final incluye una lista corta de citas APA si hay documentos usados.\n"
        "4) Si hay discrepancia entre lo que sabes y lo que dicen los documentos, dilo explícitamente y ofrece la mejor explicación.\n"
        "5) No inventes fuentes ni datos (si no estás seguro, indica el nivel de confianza o sugiere buscar/ingresar más docs).\n"
        "6) Cuando listes manifestaciones, usa viñetas '• '.\n"
        "7) Termina con una nota breve de seguridad: 'No sustituye evaluación médica'."
    )

    # Reducimos y ordenamos el contexto: pasar primero un resumen compacto de los docs
    # (tomamos las primeras N bloques del rag_text para no reventar el prompt)
    ctx_preview = "\n".join(rag_text.split("\n")[:7])  # 7 fragmentos/lineas como preview

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Contexto recuperado (fragmentos relevantes):\n{ctx_preview}"},
        {"role": "user", "content": (
            "Pregunta del usuario: " + t + "\n\n"
            "INSTRUCCIONES AL MODELO: Primero responde usando tu conocimiento general y experiencia (si aplica). "
            "Después, confirma o amplía con la evidencia local recuperada. "
            "Marca con [1], [2], ... los fragmentos del contexto cuando los uses. "
            "Si la evidencia local contradice tu conocimiento general, explica la discrepancia."
        )}
    ]

    # Ajustes de decodificación para respuestas menos 'roboticas' y más síntesis:
    out = _chat(messages, _LLM_MODEL, temperature=0.35, num_predict=350, top_p=0.95)
    raw = (out.get("message") or {}).get("content", "").strip()
    reply = _tidy_output(raw)

    # Formatea citas APA (máx 4)
    citations_apa = bm25_apa(metas)
    if reply:
        # añadimos la nota de seguridad en todas las respuestas
        if "No sustituye evaluación médica" not in reply:
            reply += " (Contenido educativo; no sustituye evaluación médica)."

    return reply or "No pude generar respuesta en este momento.", metas, citations_apa


# --- Reexportaciones (API pública del módulo) ---
upsert_document = bm25_upsert
list_docs = bm25_list
doc_stats = bm25_stats
delete_document = bm25_delete
