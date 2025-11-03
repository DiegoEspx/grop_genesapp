from __future__ import annotations
import re, logging
from typing import List, Dict, Tuple, Any
import threading
from rank_bm25 import BM25Okapi
import re

from llm_client import build_llm_from_env
from supabase_client import (
    get_prompt, get_config,
    upsert_document_chunks, 
    delete_document as sb_delete, list_documents as sb_list, get_document_stats as sb_stats,
    load_all_chunks_for_indexing,
    # Funciones de enfermedad importadas
    load_diseases, get_allowed_topics, refresh_diseases_cache 
)

# --- Configuración base ---
log = logging.getLogger("uvicorn.error")
_LLM_CLIENT, _LLM_BACKEND, _LLM_MODEL = build_llm_from_env()

# 💾 FUNCIONES DE GESTIÓN Y ADMINISTRACIÓN
# ============================================

# ... (tus funciones upsert_document, _chunk_text, etc.) ...

# Nueva función para actualizar en caliente:
def reload_all_metadata():
    """
    Función administrativa para refrescar el caché de enfermedades 
    y reconstruir el índice BM25 después de un cambio en la DB (Supabase).
    """
    log.info("Iniciando recarga manual de metadatos...")
    
    # 1. Fuerza el refresco del caché en supabase_client
    refresh_diseases_cache()
    
    # 2. Recarga los metadatos de enfermedades en memoria (_DISEASE_ALIASES, etc.)
    # El flag force_refresh=True asegura que se use la data recién traída.
    _load_disease_metadata(force_refresh=True)
    
    # 3. Reconstruye el índice BM25
    build_bm25_index()
    
    log.info("✅ Recarga de metadatos completa.")
    return True

#LÓGICA BM25 EN MEMORIA
_LOCK = threading.Lock()
_BM25_INDEX: BM25Okapi | None = None
_BM25_METAS: List[Dict] = []


def _chat(messages, model, **opts):
    return _LLM_CLIENT.chat(messages, **opts)


# --- Tokenización simple para BM25 ---
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
_TOPIC_NAMES: Dict[str, str] = {} # Para el prompt del sistema

def _load_disease_metadata(force_refresh: bool = False):
    """Carga los aliases, keywords y nombres desde la BD."""
    global _DISEASE_ALIASES, _CONDITION_SPECIFIC_KEYWORDS, _TOPIC_NAMES
    
    # load_diseases está importada de supabase_client.py
    diseases = load_diseases(force_refresh=force_refresh) 
    
    new_aliases = {}
    new_keywords = set()
    new_names = {}

    for id, data in diseases.items():
        # 1. Alias para _extract_topic (usa re.escape para evitar problemas con símbolos)
        for alias in data.get("aliases", []):
            new_aliases[rf"\b{re.escape(alias)}\b"] = id
        
        # 2. Keywords para _is_specific_rare_disease_query
        for keyword in data.get("keywords", []):
            new_keywords.add(keyword.lower())
            
        # 3. Nombres para el prompt (Paso 7 de generate_answer)
        new_names[id] = data.get("name", id)

    _DISEASE_ALIASES = new_aliases
    _CONDITION_SPECIFIC_KEYWORDS = new_keywords
    _TOPIC_NAMES = new_names
    log.info(f"✅ Metadatos de {len(diseases)} enfermedades cargados dinámicamente.")


# Inicializar los metadatos al cargar el módulo
_load_disease_metadata()

def _extract_topic(text: str) -> str | None:
    """Extrae el topic de enfermedades raras del texto."""
    t = text.lower()
    # Usa la variable global cargada dinámicamente
    for pat, topic in _DISEASE_ALIASES.items(): 
        if re.search(pat, t):
            return topic
    return None


def _is_smalltalk(t: str) -> bool:
    """Detecta si es un saludo o conversación casual."""
    t = t.lower().strip()
    # Si el mensaje es muy corto y contiene palabras de saludo
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

    # 3. ¿Combina "síndrome/enfermedad" + término médico relevante? (Reincorporado)
    if re.search(r"\b(s[ií]ndrome|enfermedad)\b", t):
        medical_terms = ["gen[ée]tic", "cromosoma", "cong[ée]nit", "raro"]
        if any(re.search(term, t) for term in medical_terms):
            return True
        
    return False


def _tidy_output(text: str, max_sentences: int = 15) -> str:
    """
    Limpia y formatea la salida del LLM, preservando el formato **Markdown**     y limitando la longitud.
    """
    # ... (Tu lógica de limpieza permanece igual) ...
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
    
    # 💡 Límite de oraciones
    sentences = re.split(r"([.!?])\s+", text)
    if len(sentences) > (max_sentences * 2):
        # Reconstruir hasta el límite
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
    """
    t = (user_msg or "").strip()

    # Obtener configuración desde BD
    llm_params = get_config("llm_params", {
        "temperature": 0.35, "max_tokens": 450, "top_p": 0.92, "max_sentences": 15
    })
    
    # 💡 Obtener los nombres de tópicos dinámicamente para mensajes
    topic_list_names = ", ".join(_TOPIC_NAMES.values())

    # 1️⃣ SMALL-TALK / SALUDO
    if _is_smalltalk(t):
        greeting = get_prompt("greeting", 
            f"¡Hola! Soy Gemi, tu asistente especializado en enfermedades raras: {topic_list_names}. ¿En qué puedo ayudarte?")
        return (greeting, [], [])

    # 2️⃣ EXTRAER Y VALIDAR TOPIC
    inferred_topic = _extract_topic(t)
    effective_topic = topic or inferred_topic
    
    # 💡 Obtener la lista de tópicos permitidos DE LA BASE DE DATOS (Soluciona "not defined")
    current_allowed_topics = get_allowed_topics()
    
    # ✅ VALIDACIÓN CRÍTICA: ¿El topic está permitido?
    if effective_topic and effective_topic not in current_allowed_topics: 
        log.warning(f"⚠️ Topic '{effective_topic}' no está en ALLOWED_TOPICS")
        effective_topic = None

    # 3️⃣ VERIFICAR SI LA CONSULTA ES DEL DOMINIO
    is_rare_disease_query = _is_specific_rare_disease_query(t)
    
    # ❌ RECHAZAR si no hay evidencia del dominio
    if not is_rare_disease_query and not effective_topic:
        out_of_scope = get_prompt("out_of_scope", 
            f"Lo siento, solo puedo ayudarte con información sobre enfermedades raras específicas: {topic_list_names}. "
            "¿Tienes alguna pregunta sobre alguna de estas condiciones?")
        log.info(f"🚫 Consulta fuera de alcance: '{t[:50]}...'")
        return (out_of_scope, [], [])

    # 4️⃣ SI NO HAY TOPIC pero SÍ es consulta médica → pedir clarificación
    if not effective_topic:
        no_topic_msg = get_prompt("no_topic", 
            f"Puedo ayudarte con información sobre {topic_list_names}. "
            "¿Podrías indicarme sobre cuál de estas condiciones te gustaría información?")
        return (no_topic_msg, [], [])

    # 5️⃣ CONSTRUIR FILTROS PARA RAG
    where = {}
    if effective_topic:
        where["topic"] = effective_topic
    if lang:
        where["lang"] = lang
    if min_year:
        where["min_year"] = min_year
    if types:
        where["type"] = types[0] if len(types) == 1 else types

    # 6️⃣ CONSULTAR BM25
    rag_params = get_config("rag_params", {"k_results": 5})
    chunks, metas = query_bm25_index(
        query_text=t,
        k=rag_params.get("k_results", 5),
        where=where if where else {}
    )

    # 7️⃣ CONSTRUIR PROMPT DEL SISTEMA CON RESTRICCIONES
    system_prompt = get_prompt("system_main", 
        "Eres Gemi, un asistente médico educativo especializado ÚNICAMENTE en enfermedades raras.")
    
    # ✅ AÑADIR RESTRICCIÓN EXPLÍCITA (Usando la lista dinámica)
    system_prompt += f"\n\n RESTRICCIÓN CRÍTICA: Solo responde preguntas sobre {topic_list_names}. Si te preguntan sobre otros temas, indica educadamente que solo puedes ayudar con estas condiciones."
    
    if effective_topic:
        # Usar el mapa de nombres dinámico
        topic_name = _TOPIC_NAMES.get(effective_topic, effective_topic)
        system_prompt += f"\n\nTEMA ACTUAL: {topic_name}"

    # PREPARAR CONTEXTO DE DOCUMENTOS
    if chunks:
        ctx_lines = []
        for i, chunk in enumerate(chunks, 1):
            ctx_lines.append(f"[{i}] {chunk['content']}")
        rag_text = "\n".join(ctx_lines)
        context_section = f"\n\nDOCUMENTOS DE REFERENCIA:\n{rag_text}"
    else:
        context_section = "\n\nNOTA: No hay documentos específicos cargados para este tema, pero usa tu conocimiento médico general SOLO sobre las condiciones permitidas."

    messages = [
        {"role": "system", "content": system_prompt + context_section},
        {"role": "user", "content": t},
    ]

    # 9️⃣ LLAMAR AL LLM
    out = _chat(
        messages, 
        _LLM_MODEL,
        temperature=llm_params.get("temperature", 0.35),
        num_predict=llm_params.get("max_tokens", 450),
        top_p=llm_params.get("top_p", 0.92)
    )
    
    raw = (out.get("message") or {}).get("content", "").strip()
    reply = _tidy_output(raw, max_sentences=llm_params.get("max_sentences", 15))

    # 🔟 FORMATEAR CITAS APA
    citations_apa = _format_apa(metas) if metas else []
    
    # 1️⃣1️⃣ NOTA DE SEGURIDAD
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
    
    # 💡 Obtener la lista de tópicos permitidos DE LA BASE DE DATOS (Soluciona "not defined")
    current_allowed_topics = get_allowed_topics()
    
    # ✅ VALIDAR TOPIC ANTES DE INSERTAR
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


# --- Reexportaciones ---
delete_document = sb_delete
list_docs = sb_list
doc_stats = sb_stats

# ============================================
# ⚙️ INICIALIZACIÓN
# ============================================
build_bm25_index()
