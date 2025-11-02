from __future__ import annotations
import re, logging
from typing import List, Dict, Tuple, Any
import threading # <-- NUEVA DEPENDENCIA
from rank_bm25 import BM25Okapi # <-- NUEVA LIBRERÍA

from llm_client import build_llm_from_env
from supabase_client import (
    get_prompt, get_config,
    # REEMPAZADO: Eliminamos 'query_documents' porque ya no la usamos
    upsert_document_chunks, 
    delete_document as sb_delete, list_documents as sb_list, get_document_stats as sb_stats,
    load_all_chunks_for_indexing # <-- NUEVA FUNCIÓN PARA BM25
)

# --- Configuración base ---
log = logging.getLogger("uvicorn.error")
_LLM_CLIENT, _LLM_BACKEND, _LLM_MODEL = build_llm_from_env() 


# ============================================
# 🧠 LÓGICA BM25 EN MEMORIA
# ============================================
_LOCK = threading.Lock()
_BM25_INDEX: BM25Okapi | None = None
_BM25_METAS: List[Dict] = []


def _chat(messages, model, **opts):
    return _LLM_CLIENT.chat(messages, **opts)


# --- Tokenización simple para BM25 ---
_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)

def _tokenize(text: str) -> List[str]:
    # ¡Importante! Debe ser la misma tokenización que se usó al guardar los tokens en Supabase.
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def build_bm25_index():
    """Carga datos de Supabase y construye el índice BM25 en memoria."""
    global _BM25_INDEX, _BM25_METAS, _LOCK
    
    log.info("🔄 Iniciando carga e índice BM25 desde Supabase...")
    
    chunks = load_all_chunks_for_indexing()
    
    if not chunks:
        log.warning("⚠️ No se encontraron chunks en Supabase para indexar. El índice estará vacío.")
        _BM25_INDEX = None
        _BM25_METAS = []
        return

    with _LOCK:
        _BM25_METAS = chunks 
        
        # Usamos los tokens que YA están guardados en el campo 'tokens' de Supabase
        tokenized_corpus = [
            chunk["tokens"] for chunk in chunks
            if chunk.get("tokens")
        ]
        
        if not tokenized_corpus:
             log.error("❌ Los chunks cargados no tienen tokens válidos para BM25.")
             return
             
        _BM25_INDEX = BM25Okapi(tokenized_corpus)
        
    log.info(f"✅ Índice BM25 construido con {len(_BM25_METAS)} chunks.")


def query_bm25_index(query_text: str, k: int, where: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Busca en el índice BM25 en memoria y aplica filtros de metadatos."""
    if _BM25_INDEX is None:
        log.error("❌ Índice BM25 no inicializado. Se intentará construir.")
        build_bm25_index() 
        if _BM25_INDEX is None:
            return [], []

    query_tokens = _tokenize(query_text)
    scores = _BM25_INDEX.get_scores(query_tokens)
    
    # Rankear por score (del más alto al más bajo)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    picked_metas = []
    
    for idx in ranked_indices:
        meta = _BM25_METAS[idx] # Obtener metadatos del chunk
        
        # Aplicar filtros de metadatos (el 'where' que viene de generate_answer)
        if where.get("topic") and meta.get("topic") != where["topic"]:
            continue
        if where.get("lang") and meta.get("lang") != where["lang"]:
            continue
        if where.get("min_year") and meta.get("year", 0) < where["min_year"]:
            continue
        # Puedes añadir otros filtros como 'type' aquí...
        
        picked_metas.append(meta)
        
        if len(picked_metas) >= k:
            break
            
    # Devuelve (chunks, metas), que en este caso son lo mismo.
    return picked_metas, picked_metas


# ============================================
# 🧬 RECONOCIMIENTO DE TEMAS Y UTILS
# ============================================
_DISEASE_ALIASES = {
# ... (dejas tus aliases aquí) ...
    r"\bdown\b": "down", r"\bwilliams\b": "williams", r"\bmps\b": "mps",
    r"\bmucopolisacaridos(?:is|es)?\b": "mps", r"\bmucopolisacaridosis\b": "mps",
    r"\bhurler\b": "mps", r"\bhunter\b": "mps", r"\bsanfilippo\b": "mps", r"\bmorquio\b": "mps",
}
# ... (el resto de tus constantes y funciones auxiliares) ...
_HEALTH_KEYWORDS = {
    "salud", "síntoma", "sintoma", "diagnóstico", "diagnostico",
    "enfermedad", "síndrome", "sindrome", "genética", "genetica",
    "tratamiento", "prevención", "prevencion", "terapia", "cuidado",
    "manifestación", "manifestacion", "característica", "caracteristica"
}

_GREET_WORDS = {
    "hola", "buenas", "buenos días", "buenas tardes", "buenas noches",
    "hey", "hi", "hello", "qué tal", "que tal"
}

_SMALLTALK_WORDS = {
    "cómo estás", "como estas", "que haces", "qué haces",
    "cómo va", "como va", "gracias", "ok", "vale"
}


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

def _is_smalltalk(t: str) -> bool:
    t = t.lower().strip()
    return any(w in t for w in _GREET_WORDS | _SMALLTALK_WORDS)

def _tidy_output(text: str, max_sentences: int = 15) -> str:
    if not text:
        return text
    text = re.sub(r"(?m)^\s*\*\s+", "• ", text)
    text = re.sub(r"([a-záéíóúñ])\s+([A-ZÁÉÍÓÚÑ])", r"\1. \2", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    sents = re.split(r"(?<=[\.\?\!])\s+", text)
    text = " ".join(sents[:max_sentences]).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


# ============================================
# 💬 GENERACIÓN DE RESPUESTA
# ============================================
def generate_answer(user_msg: str,
                    topic: str | None = None,
                    min_year: int | None = None,
                    types: list[str] | None = None,
                    lang: str | None = None):
    t = (user_msg or "").strip()

    # Obtener configuración desde BD
    llm_params = get_config("llm_params", {
        "temperature": 0.35, "max_tokens": 450, "top_p": 0.92, "max_sentences": 15
    })

    # Small-talk / saludo
    if _is_smalltalk(t):
        greeting = get_prompt("greeting", "¡Hola! Soy Gemi, tu asistente de enfermedades raras.")
        return (greeting, [], [])

    inferred = _extract_topic(t)

    # Si no hay señales médicas
    if not _is_health_related(t) and not (topic or inferred):
        no_topic_msg = get_prompt("no_topic", "¿Sobre qué tema te gustaría información?")
        return (no_topic_msg, [], [])

    effective_topic = topic or inferred
    
    # Construir filtros
    where = {}
    if effective_topic:
        where["topic"] = effective_topic
    if lang:
        where["lang"] = lang
    if min_year:
        where["min_year"] = min_year
    if types:
        where["type"] = types[0] if len(types) == 1 else types

    # Consultar documentos usando el índice BM25 en memoria
    rag_params = get_config("rag_params", {"k_results": 5})
    
    # --- CAMBIO A BM25 ---
    chunks, metas = query_bm25_index(
        query_text=t,
        k=rag_params.get("k_results", 5),
        where=where if where else {}
    )
    # --- FIN CAMBIO A BM25 ---

    # Obtener prompt del sistema desde BD
    system_prompt = get_prompt("system_main", "Eres Gemi, asistente médico educativo.")
    
    # Agregar contexto del tema
    if effective_topic:
        system_prompt += f"\n\nTEMA ACTUAL: {effective_topic}"

    # Preparar contexto de documentos
    if chunks:
        ctx_lines = []
        for i, chunk in enumerate(chunks, 1):
            ctx_lines.append(f"[{i}] {chunk['content']}")
        rag_text = "\n".join(ctx_lines)
        context_section = f"\n\nDOCUMENTOS DE REFERENCIA (úsalos para validar y complementar):\n{rag_text}"
    else:
        context_section = "\n\nNOTA: No hay documentos específicos cargados para este tema, pero puedes usar tu conocimiento médico general."

    messages = [
        {"role": "system", "content": system_prompt + context_section},
        {"role": "user", "content": t},
    ]

    # Llamar al LLM con parámetros configurables
    out = _chat(
        messages, 
        _LLM_MODEL,
        temperature=llm_params.get("temperature", 0.35),
        num_predict=llm_params.get("max_tokens", 450),
        top_p=llm_params.get("top_p", 0.92)
    )
    
    raw = (out.get("message") or {}).get("content", "").strip()
    reply = _tidy_output(raw, max_sentences=llm_params.get("max_sentences", 15))

    # Formatear citas APA
    citations_apa = _format_apa(metas) if metas else []
    
    # Nota de seguridad desde BD
    safety_note = get_prompt("safety_note", "")
    if reply and safety_note and not any(x in reply.lower() for x in ["consulta", "médico"]):
        reply += f"\n\n{safety_note}"

    return reply or "No pude generar una respuesta.", metas, citations_apa


def _format_apa(metas: List[Dict], limit: int = 4) -> List[str]:
    """Formatea metadatos en citas APA."""
    seen, out = set(), []
    for m in metas[:limit]:
        did = m.get("doc_id", "Documento")
        if did in seen:
            continue
        seen.add(did)
        
        title = m.get("title", did)
        year = m.get("year", "s.f.")
        source = m.get("source_name", "Fuente desconocida")
        
        citation = f"{title}. ({year}). {source}"
        
        if m.get("doi"):
            citation += f" DOI: {m['doi']}"
        if m.get("url"):
            citation += f" · {m['url']}"
        
        out.append(citation.strip())
    
    return out


# ============================================
# 💾 FUNCIONES DE GESTIÓN DE DOCUMENTOS
# ============================================
def upsert_document(doc_id: str, source: str, full_text: str,
                    extra_meta: Dict | None = None,
                    topic: str | None = None,
                    chunk_size: int = 1100, 
                    overlap: int = 180) -> int:
    """Inserta documento chunkeado en Supabase."""
    
    log.info(f"🔄 Iniciando upsert_document para '{doc_id}'")
    
    rag_params = get_config("rag_params", {})
    chunk_size = rag_params.get("chunk_size", chunk_size)
    overlap = rag_params.get("chunk_overlap", overlap)
    
    chunks_text = _chunk_text(full_text, chunk_size, overlap)
    
    log.info(f"✂️  Generados {len(chunks_text)} chunks de texto")
    
    if len(chunks_text) == 0:
        log.warning(f"⚠️  No se generaron chunks para '{doc_id}'")
        return 0
    
    chunks = []
    for i, content in enumerate(chunks_text):
        chunk = {
            "doc_id": doc_id,
            "topic": topic,
            **(extra_meta or {}),
            "content": content,
            "tokens": _tokenize(content),
            "chunk_index": i,
        }
        chunks.append(chunk)
    
    log.info(f"📦 Preparados {len(chunks)} chunks para Supabase")
    
    result = upsert_document_chunks(doc_id, chunks)
    log.info(f"✅ Insertados {result} chunks en Supabase")
    
    # --- NUEVO: Reconstruir el índice BM25 en memoria ---
    if result > 0:
        log.info("🔄 Actualizando índice BM25 en memoria...")
        build_bm25_index() # Llama a la nueva función
    # --- FIN NUEVO ---
    
    return result


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Divide texto en fragmentos."""
    text = " ".join((text or "").split())
    chunks, start, N = [], 0, len(text)
    
    while start < N:
        end = min(start + chunk_size, N)
        cut = end
        # Intenta cortar en un punto de puntuación, retrocediendo no más de 200 caracteres
        for sep in [". ", " ", ""]:
            idx = text.rfind(sep, start + 200, end)
            if idx != -1:
                cut = idx + (0 if sep == "" else len(sep))
                break
        
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        
        if cut >= N:
            break
        start = max(0, cut - overlap)
    
    return chunks


# --- Reexportaciones ---
delete_document = sb_delete
list_docs = sb_list
doc_stats = sb_stats

# ============================================
# ⚙️ INICIALIZACIÓN
# ============================================
# Esta línea se ejecuta al iniciar el servidor para cargar el índice por primera vez
build_bm25_index()