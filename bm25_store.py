from __future__ import annotations
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import json, os, re, threading

_LOCK = threading.Lock()
_DOCS: List[str] = []
_METAS: List[Dict] = []
_TOKENIZED: List[List[str]] = []
_BM25: BM25Okapi | None = None
_DATA_PATH = os.getenv("BM25_DATA_PATH", "bm25_docs.jsonl")

_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)

# --- Tokenización ---
def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]

# --- Reconstruir índice ---
def _rebuild_index():
    global _BM25
    _BM25 = BM25Okapi(_TOKENIZED) if _TOKENIZED else None

# --- Cargar base desde disco ---
def load_from_disk():
    if not os.path.exists(_DATA_PATH):
        return
    with _LOCK, open(_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            _DOCS.append(row["text"])
            _METAS.append(row["meta"])
            _TOKENIZED.append(_tokenize(row["text"]))
    _rebuild_index()

# --- Insertar / actualizar documento ---
def upsert_document(doc_id: str, source: str, full_text: str,
                    extra_meta: Dict | None = None,
                    topic: str | None = None,
                    chunk_size: int = 1100, overlap: int = 180,
                    replace: bool = True) -> int:
    chunks = _chunk_text(full_text, chunk_size, overlap)
    if not chunks:
        return 0

    meta_base = {"doc_id": doc_id, "source": source}
    if topic:
        meta_base["topic"] = topic
    if extra_meta:
        meta_base.update(extra_meta)

    with _LOCK, open(_DATA_PATH, "a", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            m = dict(meta_base)
            m["chunk"] = i
            _DOCS.append(c)
            _METAS.append(m)
            _TOKENIZED.append(_tokenize(c))
            f.write(json.dumps({"text": c, "meta": m}, ensure_ascii=False) + "\n")

    _rebuild_index()
    return len(chunks)

# --- Filtro tipo where (por metadatos) ---
def _match_where(meta: Dict, where: Dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_match_where(meta, w) for w in where["$and"])

    for k, cond in where.items():
        if not isinstance(cond, dict):
            if meta.get(k) != cond:
                return False
            continue
        if "$eq" in cond and meta.get(k) != cond["$eq"]:
            return False
        if "$gte" in cond:
            v = meta.get(k)
            if v is None or v < cond["$gte"]:
                return False
        if "$in" in cond:
            if meta.get(k) not in cond["$in"]:
                return False
    return True

# --- Consulta BM25 con filtros ---
def query_context(query: str, k: int = 5, where: dict | None = None) -> Tuple[str, List[Dict]]:
    if not _BM25 or not _DOCS:
        return "", []

    toks = _tokenize(query)
    scores = _BM25.get_scores(toks)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    picked, seen_doc = [], set()
    for idx, _ in ranked:
        m = _METAS[idx]
        if where and not _match_where(m, where):
            continue
        did = m.get("doc_id")
        if did in seen_doc:
            continue
        seen_doc.add(did)
        picked.append((idx, _DOCS[idx], m))
        if len(picked) >= k:
            break

    if not picked:
        return "", []

    ctx_lines, metas = [], []
    for i, (_, doc, meta) in enumerate(picked, 1):
        ctx_lines.append(f"[{i}] {doc}")
        metas.append(meta)
    return "\n".join(ctx_lines), metas

# --- Citas APA mejoradas ---
def _clean_source_name(source: str | None) -> str:
    if not source:
        return "Fuente desconocida"
    if ":" in source:
        return source.split(":", 1)[1] or source
    return source

def format_apa6_list(metas: List[Dict], limit: int = 4) -> List[str]:
    seen, out = set(), []
    for m in metas or []:
        if not isinstance(m, dict):
            continue
        did = (m.get("doc_id") or "Documento").strip()
        if did in seen:
            continue
        seen.add(did)

        title = (m.get("title") or did).strip()
        source = _clean_source_name(m.get("source"))
        year = m.get("year") or "s.f."
        tipo = f" ({m.get('type')})" if m.get("type") else ""
        country = f", {m.get('country')}" if m.get("country") else ""

        # Evita strings por defecto o valores falsos
        doi_val = m.get("doi")
        url_val = m.get("url")
        tail_parts = []
        if doi_val and isinstance(doi_val, str) and doi_val.lower() not in {"string", "none", "", "null"}:
            tail_parts.append(f"DOI: {doi_val}")
        if url_val and isinstance(url_val, str) and url_val.lower() not in {"string", "none", "", "null"}:
            tail_parts.append(url_val)
        tail_str = (" " + " · ".join(tail_parts)) if tail_parts else ""

        out.append(f"{title}. ({year}). {source}{tipo}{country}.{tail_str}".strip())

        if len(out) >= limit:
            break
    return out


# --- Eliminar documento ---
def delete_document(doc_id: str) -> int:
    if not os.path.exists(_DATA_PATH):
        return 0
    removed = 0
    with _LOCK:
        docs, metas, tokens = [], [], []
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        with open(_DATA_PATH, "w", encoding="utf-8") as f:
            for row in rows:
                m = row["meta"]
                if m.get("doc_id") == doc_id:
                    removed += 1
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                docs.append(row["text"])
                metas.append(m)
                tokens.append(_tokenize(row["text"]))
        _DOCS[:] = docs
        _METAS[:] = metas
        _TOKENIZED[:] = tokens
        _rebuild_index()
    return removed

# --- Listar documentos cargados ---
def list_docs() -> List[Dict]:
    by = {}
    for m in _METAS:
        did = m.get("doc_id", "unknown")
        by.setdefault(did, {"doc_id": did, "count_chunks": 0, "sources": set()})
        by[did]["count_chunks"] += 1
        by[did]["sources"].add(m.get("source", "unknown"))
    return sorted(
        [{"doc_id": k, "count_chunks": v["count_chunks"], "sources": sorted(v["sources"])}
         for k, v in by.items()],
        key=lambda x: x["doc_id"]
    )

# --- Obtener estadísticas de documento ---
def doc_stats(doc_id: str) -> dict:
    sample, count, sources = [], 0, set()
    for d, m in zip(_DOCS, _METAS):
        if m.get("doc_id") == doc_id:
            count += 1
            sources.add(m.get("source", "desconocido"))
            if len(sample) < 2:
                sample.append(d[:200])
    return {"doc_id": doc_id, "count": count, "sources": sorted(sources), "sample": sample}

# --- Dividir texto en fragmentos ---
def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
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

# --- Cargar índice inicial ---
load_from_disk()
