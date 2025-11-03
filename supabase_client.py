from __future__ import annotations
import os, json, logging
from typing import List, Dict, Tuple, Any

log = logging.getLogger("uvicorn.error")

# Cliente Supabase singleton
_supabase = None

def get_supabase():
    """Obtiene cliente Supabase (singleton)."""
    global _supabase
    if _supabase is None:
        try:
            from supabase import create_client, Client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            if not url or not key:
                raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY en .env")
            _supabase = create_client(url, key)
            log.info("✅ Supabase conectado correctamente")
        except ImportError:
            log.error("❌ Módulo 'supabase' no instalado. Ejecuta: pip install supabase")
            raise
        except Exception as e:
            log.error(f"❌ Error conectando a Supabase: {e}")
            raise
    return _supabase


# ============================================
# 🏥 GESTIÓN DE ENFERMEDADES
# ============================================
_diseases_cache = None  # Cache de enfermedades

def load_diseases(force_refresh: bool = False) -> Dict[str, Dict]:
    """
    Carga las enfermedades activas desde Supabase.
    Retorna un dict con estructura: {id: {name, aliases, keywords}}
    """
    global _diseases_cache
    
    # Usar cache si existe y no se fuerza refresh
    if _diseases_cache is not None and not force_refresh:
        return _diseases_cache
    
    try:
        response = get_supabase().table("diseases").select("*").eq("is_active", True).execute()
        
        if not response.data:
            log.warning("⚠️ No hay enfermedades activas en la BD. Usando valores por defecto.")
            # Fallback básico
            _diseases_cache = {
                "down": {"name": "Síndrome de Down", "aliases": ["down"], "keywords": []},
                "williams": {"name": "Síndrome de Williams", "aliases": ["williams"], "keywords": []},
                "mps": {"name": "Mucopolisacaridosis", "aliases": ["mps"], "keywords": []}
            }
            return _diseases_cache
        
        # Construir diccionario desde la BD
        diseases = {}
        for row in response.data:
            diseases[row["id"]] = {
                "name": row["name"],
                "aliases": row.get("aliases", []),
                "keywords": row.get("keywords", []),
                "description": row.get("description", "")
            }
        
        _diseases_cache = diseases
        log.info(f"🏥 {len(diseases)} enfermedades cargadas desde BD: {list(diseases.keys())}")
        return diseases
        
    except Exception as e:
        log.error(f"❌ Error cargando enfermedades: {e}")
        # Fallback en caso de error
        return {
            "down": {"name": "Síndrome de Down", "aliases": ["down"], "keywords": []},
            "williams": {"name": "Síndrome de Williams", "aliases": ["williams"], "keywords": []},
            "mps": {"name": "Mucopolisacaridosis", "aliases": ["mps"], "keywords": []}
        }


def get_allowed_topics() -> set:
    """Retorna el set de IDs de enfermedades permitidas."""
    diseases = load_diseases()
    return set(diseases.keys())


def refresh_diseases_cache():
    """Fuerza la recarga de enfermedades desde la BD."""
    global _diseases_cache
    _diseases_cache = None
    return load_diseases(force_refresh=True)


# ============================================
# PROMPTS
# ============================================
_prompt_cache = {}  # Cache manual

def get_prompt(name: str, default: str = "") -> str:
    """Obtiene un prompt de la BD."""
    if name in _prompt_cache:
        return _prompt_cache[name]
    
    try:
        response = get_supabase().table("prompts").select("content").eq("name", name).eq("is_active", True).single().execute()
        if response.data:
            content = response.data["content"]
            _prompt_cache[name] = content
            log.info(f"📝 Prompt '{name}' cargado desde BD")
            return content
        log.warning(f"⚠️  Prompt '{name}' no encontrado, usando default")
        return default
    except Exception as e:
        log.warning(f"⚠️  Error obteniendo prompt '{name}': {e}. Usando default")
        return default


def update_prompt(name: str, content: str) -> bool:
    """Actualiza un prompt en la BD."""
    try:
        get_supabase().table("prompts").update({
            "content": content,
        }).eq("name", name).execute()
        if name in _prompt_cache:
            del _prompt_cache[name]
        log.info(f"✅ Prompt '{name}' actualizado")
        return True
    except Exception as e:
        log.error(f"❌ Error actualizando prompt '{name}': {e}")
        return False


# ============================================
# CONFIGURACIÓN
# ============================================
_config_cache = {}

def get_config(key: str, default: Any = None) -> Any:
    """Obtiene configuración de la BD."""
    if key in _config_cache:
        return _config_cache[key]
    
    try:
        response = get_supabase().table("config").select("value").eq("key", key).single().execute()
        if response.data:
            value = response.data["value"]
            _config_cache[key] = value
            log.info(f"⚙️  Config '{key}' cargada desde BD")
            return value
        log.warning(f"⚠️  Config '{key}' no encontrada, usando default")
        return default
    except Exception as e:
        log.warning(f"⚠️  Error obteniendo config '{key}': {e}. Usando default")
        return default


def update_config(key: str, value: Any) -> bool:
    """Actualiza configuración en la BD."""
    try:
        get_supabase().table("config").upsert({
            "key": key,
            "value": value
        }).execute()
        if key in _config_cache:
            del _config_cache[key]
        log.info(f"✅ Config '{key}' actualizada")
        return True
    except Exception as e:
        log.error(f"❌ Error actualizando config '{key}': {e}")
        return False


# ============================================
# DOCUMENTOS
# ============================================
def upsert_document_chunks(doc_id: str, chunks: List[Dict]) -> int:
    """Inserta/actualiza chunks de documento en Supabase."""
    try:
        get_supabase().table("documents").delete().eq("doc_id", doc_id).execute()
        log.info(f"🗑️  Chunks antiguos de '{doc_id}' eliminados")
        
        if not chunks:
            return 0
        
        rows = []
        for chunk in chunks:
            row = {
                "doc_id": doc_id,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "tokens": chunk["tokens"],
                "topic": chunk.get("topic"),
                "lang": chunk.get("lang", "es"),
                "year": chunk.get("year"),
                "type": chunk.get("type"),
                "country": chunk.get("country"),
                "title": chunk.get("title"),
                "doi": chunk.get("doi"),
                "url": chunk.get("url"),
                "source_name": chunk.get("source_name"),
            }
            rows.append(row)
        
        get_supabase().table("documents").insert(rows).execute()
        log.info(f"✅ {len(rows)} chunks de '{doc_id}' insertados en Supabase")
        
        return len(rows)
    except Exception as e:
        log.error(f"❌ Error insertando documento '{doc_id}': {e}")
        return 0


def delete_document(doc_id: str) -> int:
    """Elimina documento de Supabase."""
    try:
        response = get_supabase().table("documents").delete().eq("doc_id", doc_id).execute()
        count = len(response.data) if response.data else 0
        log.info(f"🗑️  {count} chunks de '{doc_id}' eliminados")
        return count
    except Exception as e:
        log.error(f"❌ Error eliminando documento '{doc_id}': {e}")
        return 0


def list_documents() -> List[Dict]:
    """Lista todos los documentos únicos."""
    try:
        response = get_supabase().table("documents").select("doc_id, title, topic, source_name, created_at").execute()
        
        docs_map = {}
        for row in response.data:
            did = row["doc_id"]
            if did not in docs_map:
                docs_map[did] = {
                    "doc_id": did,
                    "title": row.get("title", did),
                    "topic": row.get("topic"),
                    "source": row.get("source_name"),
                    "created_at": row.get("created_at"),
                    "chunk_count": 0
                }
            docs_map[did]["chunk_count"] += 1
        
        log.info(f"📋 {len(docs_map)} documentos listados")
        return list(docs_map.values())
    except Exception as e:
        log.error(f"❌ Error listando documentos: {e}")
        return []


def get_document_stats(doc_id: str) -> Dict:
    """Obtiene estadísticas de un documento."""
    try:
        response = get_supabase().table("documents").select("*").eq("doc_id", doc_id).execute()
        
        if not response.data:
            return {"doc_id": doc_id, "found": False}
        
        chunks = response.data
        return {
            "doc_id": doc_id,
            "found": True,
            "chunk_count": len(chunks),
            "topic": chunks[0].get("topic"),
            "title": chunks[0].get("title"),
            "sample": chunks[0]["content"][:200] if chunks else ""
        }
    except Exception as e:
        log.error(f"❌ Error obteniendo stats de '{doc_id}': {e}")
        return {"doc_id": doc_id, "error": str(e)}
    

def load_all_chunks_for_indexing() -> List[Dict]:
    """Obtiene todos los chunks y sus metadatos de Supabase."""
    try:
        response = get_supabase().table("documents").select(
            "doc_id, chunk_index, content, tokens, topic, lang, year, title"
        ).execute()
        
        data = response.data or []
        log.info(f"📚 {len(data)} chunks cargados de Supabase para indexación.")
        return data
    except Exception as e:
        log.error(f"❌ Error cargando chunks para BM25: {e}")
        return []