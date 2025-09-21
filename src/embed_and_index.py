#!/usr/bin/env python3
import os, json
from pathlib import Path
import faiss
import numpy as np

# prefer Ollama HTTP only if you need; here we use the python client
import ollama

INPUT_ROOT = Path("output")           # scans output/SET*
INDEX_DIR  = Path("vector_store/all") # single combined index for all sets
MODEL      = "nomic-embed-text"       # pull with: ollama pull nomic-embed-text

def iter_rows(root: Path):
    for setdir in sorted(root.glob("SET*")):
        # prefer single file
        qfile = setdir / "questions.jsonl"
        if qfile.exists():
            with qfile.open("r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    yield {
                        "id": r.get("id"),
                        "text": r.get("text","").strip(),
                        "chapter": r.get("chapter") or "Unknown",
                        "paper_id": r.get("paper_id") or setdir.name,
                        "fmt": r.get("format") or "GENERAL",
                        "section": r.get("section_heading") or "",
                        "_set": setdir.name,
                    }
        else:
            # subfolder files fallback
            for p in setdir.glob("*/questions_part*.jsonl"):
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        r["_set"] = setdir.name
                        yield r

def embed_one(text: str):
    if not text:
        return None
    e = ollama.embeddings(model=MODEL, prompt=text)["embedding"]
    return np.array(e, dtype="float32")

def main():
    rows = list(iter_rows(INPUT_ROOT))
    if not rows:
        raise SystemExit(f"No questions found under: {INPUT_ROOT.resolve()}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = INDEX_DIR / "meta.jsonl"
    index_path = INDEX_DIR / "index.faiss"

    vecs, metas = [], []
    for r in rows:
        v = embed_one(r["text"])
        if v is None:
            continue
        metas.append(r)
        vecs.append(v)

    if not vecs:
        raise SystemExit("No vectors generated — check OCR/extraction output.")

    mat = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[ok] wrote index: {index_path}  (N={len(metas)}, dim={mat.shape[1]})")
    print(f"[ok] wrote meta:  {meta_path}")

if __name__ == "__main__":
    main()
