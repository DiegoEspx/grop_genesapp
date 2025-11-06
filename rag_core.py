from __future__ import annotations
import re, logging
from typing import List, Dict, Tuple, Any
import threading
from rank_bm25 import BM25Okapi
from llm_client import build_llm_from_env
from supabase_client import (
    get_prompt, get_config,
    upsert_document_chunks, 
    delete_document as sb_delete, list_documents as sb_list, get_document_stats as sb_stats,
    load_all_chunks_for_indexing,
    load_diseases, get_allowed_topics, refresh_diseases_cache 
)

# --- Configuración base ---
log = logging.getLogger("uvicorn.error")
_LLM_CLIENT, _LLM_BACKEND, _LLM_MODEL = build_llm_from_env()

# FUNCIONES DE GESTIÓN Y ADMINISTRACIÓN

_LOCK = threading.Lock()
_BM25_INDEX: BM25Okapi | None = None
_BM25_METAS: List[Dict] = []

def _chat(messages, model, **opts):
    return _LLM_CLIENT.chat(messages, **opts)

_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)

def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]

def build_bm25_index():
    """Carga datos de Supabase y construye el índice BM25 en memoria."""
    global _BM25_INDEX, _BM25_METAS, _LOCK
    
    log.info("Iniciando carga e índice BM25 desde Supabase...")
    
    chunks = load_all_chunks_for_indexing()
    
    if not chunks:
        log.warning("No se encontraron chunks en Supabase para indexar. El índice estará vacío.")
        _BM25_INDEX = None
        _BM25_METAS = []
        return

    with _LOCK:
        _BM25_METAS = chunks 
        tokenized_corpus = [
            chunk["tokens"] for chunk in chunks
            if chunk.get("tokens")
        ]
        
        if not tokenized_corpus:
                log.error("Los chunks cargados no tienen tokens válidos para BM25.")
                return
        _BM25_INDEX = BM25Okapi(tokenized_corpus)
        
    log.info(f"Índice BM25 construido con {len(_BM25_METAS)} chunks.")

def query_bm25_index(query_text: str, k: int, where: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Busca en el índice BM25 en memoria y aplica filtros de metadatos."""
    if _BM25_INDEX is None:
        log.error("Índice BM25 no inicializado. Se intentará construir.")
        build_bm25_index() 
        if _BM25_INDEX is None:
            return [], []

    query_tokens = _tokenize(query_text)
    scores = _BM25_INDEX.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    picked_metas = []
    
    for idx in ranked_indices:
        meta = _BM25_METAS[idx]
        
        # Aplicar filtros de metadatos
        if where.get("topic") and meta.get("topic") != where["topic"]:
            continue
        if where.get("lang") and meta.get("lang") != where["lang"]:
            continue
        if where.get("min_year") and meta.get("year", 0) < where["min_year"]:
            continue
        
        picked_metas.append(meta)
        
        if len(picked_metas) >= k:
            break
            
    return picked_metas, picked_metas

_GREET_WORDS = {
    "hola", "buenas", "buenos días", "buenas tardes", "buenas noches",
    "hey", "hi", "hello", "qué tal", "que tal", "saludos"
}

_SMALLTALK_WORDS = {
    "cómo estás", "como estas", "que haces", "qué haces",
    "cómo va", "como va", "gracias", "ok", "vale", "genial"
}

# --- LÓGICA DINÁMICA DE ENFERMEDADES ---
_DISEASE_ALIASES: Dict[str, str] = {}
_CONDITION_SPECIFIC_KEYWORDS: set[str] = set()
_TOPIC_NAMES: Dict[str, str] = {} 

from supabase_client import load_diseases

def _load_disease_metadata(force_refresh: bool = False):
    """
    Carga los metadatos de enfermedades EXCLUSIVAMENTE desde la tabla 'diseases' en Supabase.
    """
    global DISEASES_CACHE

    diseases = load_diseases(force_refresh=force_refresh)
    DISEASES_CACHE = {}

    for d in diseases.values():
        if not d.get("is_active", True):
            continue

        disease_id = d["id"]
        aliases = [a.lower() for a in d.get("aliases", [])]
        keywords = [k.lower() for k in d.get("keywords", [])]
        name = d["name"].lower()

        DISEASES_CACHE[disease_id] = {
            "name": name,
            "aliases": aliases,
            "keywords": keywords
        }

    log.info(f"✅ Cargadas {len(DISEASES_CACHE)} enfermedades desde Supabase.")


def _extract_topic(text: str) -> str | None:
    """Extrae el topic de enfermedades raras del texto."""
    t = text.lower()
    for pat, topic in _DISEASE_ALIASES.items(): 
        if re.search(pat, t):
            return topic
    return None


def _is_smalltalk(t: str) -> bool:
    """Detecta si es un saludo o conversación casual."""
    t = t.lower().strip()
    words = set(t.split())
    if len(words) <= 5 and any(w in _GREET_WORDS | _SMALLTALK_WORDS for w in words):
        return True
    return False


def _is_specific_rare_disease_query(text: str) -> bool:
    """
    Verifica si la consulta es ESPECÍFICAMENTE sobre las enfermedades raras soportadas.
    Retorna True SOLO si hay evidencia clara del dominio.
    """
    t = text.lower()
    # 1. ¿Menciona explícitamente alguna condición?
    if _extract_topic(t):
        return True
    
    # 2. ¿Contiene términos médicos específicos de estas condiciones?
    if any(kw in t for kw in _CONDITION_SPECIFIC_KEYWORDS): 
        return True

    # 3. ¿Combina "síndrome/enfermedad" + término médico relevante?
    if re.search(r"\b(s[ií]ndrome|enfermedad)\b", t):
        medical_terms = ["gen[ée]tic", "cromosoma", "cong[ée]nit", "raro"]
        if any(re.search(term, t) for term in medical_terms):
            return True
        
    return False


def _tidy_output(text: str, max_sentences: int = 15) -> str:
    """
    Limpia y formatea la salida del LLM, preservando el formato **Markdown**     y limitando la longitud.
    """
    if not text:
        return text
        
    text = re.sub(r"\*\*([^*]+?)\:\s*\*\*", r"\*\* \1\*\*", text)
    text = re.sub(r"\*\*([^*]+)\:\s*([^\*])", r"**\1** \2", text)
    text = re.sub(r"\*\*([^*]+)\*\*::", r"**\1**", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^\s*[\*\-]\s+", "* ", text, flags=re.MULTILINE)
    text = re.sub(r"([a-z])\s*\n([A-ZÁÉÍÓÚÑ]{2,}[^a-z])", r"\1\n\n\2", text)
    text = re.sub(r"([^\n])\n(\*)", r"\1\n\n\2", text)
    text = re.sub(r":\s*(\*)", r":\n\n\1", text)
    text = re.sub(r"([a-záéíóúüñ])\.\s+([A-ZÁÉÍÓÚÑ])", r"\1 \2", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"^\s*\.\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    
    sentences = re.split(r"([.!?])\s+", text)
    if len(sentences) > (max_sentences * 2):
        text = "".join(sentences[:max_sentences * 2 - 1]) 
    
    if text and text[-1] not in ".!?\n":
        text += "."
    
    return text

# GENERACIÓN DE RESPUESTA CON LÍMITES ESTRICTOS
def generate_answer(user_msg: str,
                    topic: str | None = None,
                    min_year: int | None = None,
                    types: list[str] | None = None,
                    lang: str | None = None):
    """
    Genera respuesta con validación estricta del dominio.
    Usa DISEASES_CACHE, _TOPIC_NAMES y _DISEASE_ALIASES cargadas desde Supabase.
    """
    t = (user_msg or "").strip()

    # Configuración dinámica
    llm_params = get_config("llm_params", {
        "temperature": 0.35, "max_tokens": 450, "top_p": 0.92, "max_sentences": 15
    })
    
    # Lista de nombres de topics para prompts
    topic_list_names = ", ".join(_TOPIC_NAMES.values())

    # Small talk
    if _is_smalltalk(t):
        greeting = get_prompt("greeting", 
            f"¡Hola! Soy Gemi, tu asistente especializado en enfermedades raras: {topic_list_names}. ¿En qué puedo ayudarte?")
        return greeting, [], []

    # Extraer topic de texto y validar contra allowed
    inferred_topic = _extract_topic(t)
    effective_topic = topic or inferred_topic
    current_allowed_topics = get_allowed_topics()
    if effective_topic and effective_topic not in current_allowed_topics:
        log.warning(f"⚠️ Topic '{effective_topic}' no está permitido")
        effective_topic = None

    # Verificar si es consulta de dominio
    is_rare_disease_query = _is_specific_rare_disease_query(t)
    if not is_rare_disease_query and not effective_topic:
        out_of_scope = get_prompt("out_of_scope", 
            f"Lo siento, solo puedo ayudarte con información sobre enfermedades raras específicas: {topic_list_names}.")
        log.info(f"🚫 Consulta fuera de alcance: '{t[:50]}...'")
        return out_of_scope, [], []

    # Pedir clarificación si no hay topic
    if not effective_topic:
        no_topic_msg = get_prompt("no_topic", 
            f"Puedo ayudarte con información sobre {topic_list_names}. ¿Podrías indicarme cuál te interesa?")
        return no_topic_msg, [], []

    # Construir filtros para BM25
    where = {}
    if effective_topic: where["topic"] = effective_topic
    if lang: where["lang"] = lang
    if min_year: where["min_year"] = min_year
    if types: where["type"] = types[0] if len(types) == 1 else types

    # Consultar BM25
    rag_params = get_config("rag_params", {"k_results": 5})
    chunks, metas = query_bm25_index(
        query_text=t,
        k=rag_params.get("k_results", 5),
        where=where if where else {}
    )

    # Construir prompt del sistema
    system_prompt = get_prompt("system_main", 
        "Eres Gemi, asistente educativo especializado ÚNICAMENTE en enfermedades raras.")
    system_prompt += f"\n\nRESTRICCIÓN: Solo responde sobre {topic_list_names}."
    if effective_topic:
        topic_name = _TOPIC_NAMES.get(effective_topic, effective_topic)
        system_prompt += f"\nTEMA ACTUAL: {topic_name}"

    # Contexto de documentos
    if chunks:
        context_section = "\n\nDOCUMENTOS DE REFERENCIA:\n" + "\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks))
    else:
        context_section = "\n\nNOTA: No hay documentos cargados, usa solo tu conocimiento médico sobre las condiciones permitidas."

    messages = [
        {"role": "system", "content": system_prompt + context_section},
        {"role": "user", "content": t},
    ]

    # Llamar LLM
    out = _chat(messages, _LLM_MODEL,
                temperature=llm_params.get("temperature", 0.35),
                num_predict=llm_params.get("max_tokens", 450),
                top_p=llm_params.get("top_p", 0.92))
    
    raw = (out.get("message") or {}).get("content", "").strip()
    reply = _tidy_output(raw, max_sentences=llm_params.get("max_sentences", 15))

    # Formatear citas APA
    citations_apa = _format_apa(metas) if metas else []

    # Nota de seguridad
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

# FUNCIONES DE GESTIÓN DE DOCUMENTOS
def upsert_document(doc_id: str, source: str, full_text: str,
                    extra_meta: Dict | None = None,
                    topic: str | None = None,
                    chunk_size: int = 1100, 
                    overlap: int = 180) -> int:
    """Inserta documento chunkeado en Supabase."""

    current_allowed_topics = get_allowed_topics()
    
    # VALIDAR TOPIC ANTES DE INSERTAR
    if topic and topic not in current_allowed_topics:
        log.error(f"❌ Topic '{topic}' no permitido. Solo se aceptan: {current_allowed_topics}")
        return 0
    
    log.info(f"🔄 Iniciando upsert_document para '{doc_id}'")
    
    rag_params = get_config("rag_params", {})
    chunk_size = rag_params.get("chunk_size", chunk_size)
    overlap = rag_params.get("chunk_overlap", overlap)
    
    chunks_text = _chunk_text(full_text, chunk_size, overlap)
    
    log.info(f"✂️  Generados {len(chunks_text)} chunks de texto")
    
    if len(chunks_text) == 0:
        log.warning(f"⚠️  No se generaron chunks para '{doc_id}'")
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
    
    if result > 0:
        log.info("🔄 Actualizando índice BM25 en memoria...")
        build_bm25_index()
    
    return result

def reload_all_metadata():
    refresh_diseases_cache()
    _load_disease_metadata(force_refresh=True)
    build_bm25_index()
    log.info("✅ Recarga de metadatos completa.")
    return True

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Divide texto en fragmentos."""
    text = " ".join((text or "").split())
    chunks, start, N = [], 0, len(text)
    
    while start < N:
        end = min(start + chunk_size, N)
        cut = end
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

delete_document = sb_delete
list_docs = sb_list
doc_stats = sb_stats

build_bm25_index()