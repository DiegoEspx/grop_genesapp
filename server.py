from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from io import BytesIO
from dotenv import load_dotenv

load_dotenv(override=True)

from rag_core import (
    upsert_document,
    generate_answer,
    delete_document,
    doc_stats,
    list_docs,
    reload_all_metadata,
)

app = FastAPI(title="Groq RAG Server", docs_url="/swagger")

# Configurar CORS para permitir conexiones desde Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- MODELOS --------
class ChatRequest(BaseModel):
    message: str
    topic: str | None = None
    lang: str | None = "es"


class ChatResponse(BaseModel):
    reply: str
    citations: list | None = None
    citations_apa: list | None = None


# -------- ENDPOINTS DE ADMINISTRACIÓN DE DATOS --------


@app.post("/admin/reload-metadata")
def admin_reload():
    """
    Endpoint de administración.
    Recarga los metadatos de enfermedades y reconstruye el índice RAG
    después de cambios manuales en Supabase (ej: añadir la enfermedad de Turner).
    """
    import logging

    log = logging.getLogger("uvicorn.error")

    log.info("📢 Solicitud de recarga manual recibida.")

    try:
        success = reload_all_metadata()
        if success:
            return {
                "status": "success",
                "message": "Metadatos y RAG index recargados en caliente (Hot Reload).",
            }
        else:
            return {
                "status": "error",
                "message": "La función de recarga terminó sin éxito.",
            }
    except Exception as e:
        log.error(f"❌ Error al recargar metadatos: {e}")
        import traceback

        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error interno durante la recarga: {str(e)}",
        }


# -------- ENDPOINTS GENERALES --------


@app.get("/health")
def health():
    return {"status": "ok", "backend": "groq", "storage": "supabase"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Chat con el asistente Gemi."""
    topic = req.topic if hasattr(req, "topic") else None
    lang = req.lang if hasattr(req, "lang") else "es"

    reply, metas, apa = generate_answer(
        req.message,
        topic=topic,
        lang=lang,
    )
    return ChatResponse(reply=reply, citations=metas, citations_apa=apa)


@app.post("/ingest")
def ingest(
    file: UploadFile = File(...),
    doc_id: str | None = Form(None),
    topic: str | None = Form(None),
    lang: str | None = Form(None),
    year: int | None = Form(None),
    type: str | None = Form(None),
    country: str | None = Form(None),
    title: str | None = Form(None),
    doi: str | None = Form(None),
    url: str | None = Form(None),
):
    """Ingesta de documentos PDF a Supabase."""
    import logging

    log = logging.getLogger("uvicorn.error")
    name = file.filename or "unknown"
    doc_id = doc_id or name
    raw = file.file.read()

    log.info(f"📄 Procesando PDF: {name} ({len(raw)} bytes)")

    try:
        pages = PdfReader(BytesIO(raw)).pages
        log.info(f"📖 PDF tiene {len(pages)} páginas")

        text = "\n".join([(p.extract_text() or "") for p in pages])
        log.info(f"📝 Texto extraído: {len(text)} caracteres")

        if len(text.strip()) == 0:
            return {
                "ok": False,
                "error": "El PDF no contiene texto extraíble. Puede ser un PDF de solo imágenes (necesitas OCR).",
            }

    except Exception as e:
        log.error(f"❌ Error leyendo PDF: {e}")
        return {"ok": False, "error": f"Error leyendo PDF: {str(e)}"}

    def _clean(v):
        if v is None:
            return None
        s = str(v).strip().lower()
        return None if s in {"", "string", "none", "null"} else v

    meta = {
        "topic": _clean(topic),
        "lang": _clean(lang) or "es",  # default español
        "year": year if year and year > 1900 else None,
        "type": _clean(type),
        "country": _clean(country),
        "title": _clean(title) or doc_id,
        "doi": _clean(doi),
        "url": _clean(url),
        "source_name": name,
    }

    meta = {k: v for k, v in meta.items() if v is not None}

    log.info(f"📋 Metadatos: {meta}")
    try:
        count = upsert_document(
            doc_id=doc_id,
            source=f"upload:{name}",
            full_text=text,
            extra_meta=meta,
            topic=meta.get("topic"),
        )

        log.info(f"✅ Documento '{doc_id}' indexado: {count} chunks")

        return {
            "ok": True,
            "chunks_indexed": count,
            "doc_id": doc_id,
            "storage": "supabase",
            "metadata": meta,
            "text_length": len(text),
            "pages": len(pages),
        }
    except Exception as e:
        log.error(f"❌ Error en upsert_document: {e}")
        import traceback

        traceback.print_exc()
        return {"ok": False, "error": str(e)}


@app.get("/docs")
def docs_list():
    """Lista todos los documentos en Supabase."""
    return {"items": list_docs(), "storage": "supabase"}


@app.get("/docs/{doc_id}")
def docs_info(doc_id: str):
    """Información detallada de un documento."""
    return doc_stats(doc_id)


@app.delete("/docs/{doc_id}")
def docs_delete(doc_id: str):
    """Elimina un documento de Supabase."""
    try:
        deleted = delete_document(doc_id)
        return {"ok": True, "deleted": deleted, "doc_id": doc_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -------- ENDPOINTS PARA GESTIÓN DE PROMPTS Y CONFIGURACIÓN --------
from supabase_client import get_prompt, update_prompt, get_config, update_config


@app.get("/admin/prompts")
def get_prompts():
    """Obtiene todos los prompts disponibles."""
    from supabase_client import get_supabase

    try:
        response = get_supabase().table("prompts").select("*").execute()
        return {"prompts": response.data}
    except Exception as e:
        return {"error": str(e)}


@app.get("/admin/prompts/{name}")
def get_prompt_by_name(name: str):
    """Obtiene un prompt específico."""
    content = get_prompt(name)
    return {"name": name, "content": content}


@app.put("/admin/prompts/{name}")
def update_prompt_endpoint(name: str, content: str = Form(...)):
    """Actualiza un prompt."""
    success = update_prompt(name, content)
    return {"ok": success, "name": name}


@app.get("/admin/config")
def get_all_config():
    """Obtiene toda la configuración."""
    from supabase_client import get_supabase

    try:
        response = get_supabase().table("config").select("*").execute()
        return {"config": response.data}
    except Exception as e:
        return {"error": str(e)}


@app.get("/admin/config/{key}")
def get_config_endpoint(key: str):
    """Obtiene un valor de configuración."""
    value = get_config(key)
    return {"key": key, "value": value}


@app.put("/admin/config/{key}")
def update_config_endpoint(key: str, value: str = Form(...)):
    """Actualiza configuración."""
    import json

    try:
        parsed = json.loads(value)
        success = update_config(key, parsed)
    except json.JSONDecodeError:
        success = update_config(key, value)

    return {"ok": success, "key": key}
