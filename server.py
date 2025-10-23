from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pypdf import PdfReader
from io import BytesIO
from pydantic import BaseModel

from rag_core import (
    upsert_document, generate_answer,
    delete_document, doc_stats, list_docs,
)

app = FastAPI(title="Groq RAG Server", docs_url="/swagger")

class ChatRequest(BaseModel):
    message: str
    topic: str | None = None

class ChatResponse(BaseModel):
    reply: str
    citations: list | None = None
    citations_apa: list | None = None

@app.get("/health")
def health():
    return {"status": "ok", "backend": "groq"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply, metas, apa = generate_answer(
        req.message,
        topic=req.topic,          
        lang=req.lang,
    )
    return ChatResponse(reply=reply, citations=metas, citations_apa=apa)

@app.post("/ingest")
def ingest(
    file: UploadFile = File(...),
    doc_id: str = Form(None),
    topic: str = Form(None),
    year: int = Form(None),
    type: str = Form(None),
    lang: str = Form(None),
    country: str = Form(None),
    doi: str = Form(None),
    url: str = Form(None),
    title: str = Form(None),
):
    name = file.filename or "unknown"
    doc_id = doc_id or name
    raw = file.file.read()
    reader = PdfReader(BytesIO(raw))
    text = "\n".join([(p.extract_text() or "") for p in reader.pages]).strip()
    if not text:
        raise HTTPException(status_code=422, detail="No se extrajo texto (¿PDF escaneado sin OCR?).")

    # metadatos enriquecidos
    extra = {
        "topic": topic,
        "year": year,
        "type": type,
        "lang": lang,
        "country": country,
        "doi": doi,
        "url": url,
        "title": title or name,  # mostrar título/archivo en citas
        "source_name": name,     # para mostrar siempre el nombre del archivo
    }
    # limpia None
    extra = {k: v for k, v in extra.items() if v is not None}

    count = upsert_document(
        doc_id=doc_id,
        source=f"upload:{name}",
        full_text=text,
        extra_meta=extra,
        topic=topic,
        replace=True,
    )
    return {"ok": True, "chunks_indexed": count, "doc_id": doc_id, "meta": extra}


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
