from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from pypdf import PdfReader
from io import BytesIO
from dotenv import load_dotenv

from rag_core import (
    upsert_document, generate_answer,
    delete_document, doc_stats, list_docs,
)

load_dotenv(override=True)

app = FastAPI(title="Groq RAG Server", docs_url="/swagger")

# -------- MODELOS --------
class ChatRequest(BaseModel):
    message: str
    topic: str | None = None       
    lang: str | None = "es"        

class ChatResponse(BaseModel):
    reply: str
    citations: list | None = None
    citations_apa: list | None = None

# -------- ENDPOINTS --------
@app.get("/health")
def health():
    return {"status": "ok", "backend": "groq"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # acceso tolerante por si alguna versión antigua del cliente no manda lang/topic
    topic = (req.topic if hasattr(req, "topic") else None)
    lang  = (req.lang  if hasattr(req, "lang")  else "es")

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
    name = file.filename or "unknown"
    doc_id = doc_id or name
    raw = file.file.read()
    text = "\n".join([(p.extract_text() or "") for p in PdfReader(BytesIO(raw)).pages])

    # limpia placeholders típicos de Swagger
    def _clean(v):
        if v is None: return None
        s = str(v).strip().lower()
        return None if s in {"", "string", "none", "null"} else v

    meta = {
        "topic": _clean(topic),
        "lang": _clean(lang),
        "year": year,
        "type": _clean(type),
        "country": _clean(country),
        "title": _clean(title) or doc_id,
        "doi": _clean(doi),
        "url": _clean(url),
        "source_name": name,
    }
    # quita None
    meta = {k: v for k, v in meta.items() if v is not None}

    count = upsert_document(
        doc_id=doc_id,
        source=f"upload:{name}",
        full_text=text,
        extra_meta=meta,
        topic=meta.get("topic"),
    )
    return {"ok": True, "chunks_indexed": count, "doc_id": doc_id}

@app.get("/docs")
def docs_list():
    return {"items": list_docs()}

@app.get("/docs/{doc_id}")
def docs_info(doc_id: str):
    return doc_stats(doc_id)

@app.delete("/docs/{doc_id}")
def docs_delete(doc_id: str):
    deleted = delete_document(doc_id)
    return {"ok": True, "deleted": deleted, "doc_id": doc_id}
