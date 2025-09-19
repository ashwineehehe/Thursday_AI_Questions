#!/usr/bin/env python3
import os, json, glob
from pathlib import Path
import faiss
import numpy as np
import ollama  # pip install ollama

# CONFIG
OUTPUT_ROOT = Path("output/SET1")         # where important_questions.py wrote folders
MODEL = "nomic-embed-text"                # or "bge-m3"
INDEX_DIR = Path("vector_store/SET1")     # where to persist FAISS + metadata

def iter_question_rows(base: Path):
    # also allow questions.jsonl directly in base
    base_file = base / "questions.jsonl"
    if base_file.exists():
        with base_file.open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    # otherwise, check subfolders
    for paper_dir in base.iterdir():
        if not paper_dir.is_dir():
            continue
        files = list(paper_dir.glob("questions.jsonl"))
        if not files:
            files = sorted(paper_dir.glob("questions_part*.jsonl"))
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)

def embed_batch(texts, model=MODEL):
    """
    Embeds a list of strings via Ollama.
    Uses python client; alternatively you can call the HTTP API.
    """
    # ollama.embeddings() supports one prompt at a time — do a simple loop
    embs = []
    for t in texts:
        if not t:
            embs.append(None)
            continue
        e = ollama.embeddings(model=model, prompt=t)["embedding"]
        embs.append(e)
    return embs

def build_faiss_index(rows, index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "meta.jsonl"
    index_path = index_dir / "index.faiss"

    # stream rows, embed in small batches
    BATCH = 32
    buff_rows, buff_texts = [], []
    all_vecs = []
    metas = []

    def flush():
        nonlocal all_vecs, metas, buff_rows, buff_texts
        if not buff_rows:
            return
        embs = embed_batch(buff_texts)
        for r, e in zip(buff_rows, embs):
            if e is None: 
                continue
            metas.append(r)
            all_vecs.append(np.array(e, dtype="float32"))
        buff_rows, buff_texts = [], []

    for r in rows:
        # minimal filtering
        if not r["text"] or len(r["text"].split()) < 4:
            continue
        buff_rows.append(r)
        buff_texts.append(r["text"])
        if len(buff_rows) >= BATCH:
            flush()
    flush()

    if not all_vecs:
        raise SystemExit("No vectors generated — check your input/output paths and OCR results.")

    mat = np.vstack(all_vecs)  # (N, D)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)    # cosine via inner product if normalized; we’ll L2 normalize:
    faiss.normalize_L2(mat)

    index.add(mat)

    # persist FAISS
    faiss.write_index(index, str(index_path))

    # persist metadata aligned to index order
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[ok] wrote index: {index_path} (vectors={len(metas)}, dim={dim})")
    print(f"[ok] wrote meta:  {meta_path}")

def main():
    rows = list(iter_question_rows(OUTPUT_ROOT))
    if not rows:
        raise SystemExit(f"No questions found under: {OUTPUT_ROOT.resolve()}\n"
                         "Make sure important_questions.py wrote questions.jsonl or questions_part*.jsonl")
    print(f"[data] found {len(rows)} question rows")
    build_faiss_index(rows, INDEX_DIR)

if __name__ == "__main__":
    main()