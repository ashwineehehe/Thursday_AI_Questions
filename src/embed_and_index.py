# NEW imports/args
import argparse, time, hashlib, numpy as np, faiss, json
from pathlib import Path
import ollama, datetime

def h(txt): return hashlib.sha1(txt.strip().lower().encode('utf-8')).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--set-id", required=True)  # e.g., SET1 or ALL
parser.add_argument("--emb-model", default="nomic-embed-text")
args = parser.parse_args()

EMB_MODEL = args.emb_model

def embed_batch(texts):
    # simple serial batch; adjust if you wrap multiple requests
    vecs=[]
    for t in texts:
        e = ollama.embeddings(model=EMB_MODEL, prompt=t)["embedding"]
        vecs.append(np.array(e, dtype="float32"))
        time.sleep(0.02)  # gentle throttle
    return np.vstack(vecs)

def build_for_set(set_id):
    OUTPUT_DIR = Path("output")/set_id
    INDEX_DIR  = Path("vector_store")/set_id
    qfile = OUTPUT_DIR/"questions.jsonl"
    if not qfile.exists():
        print(f"[skip] no questions.jsonl for {set_id}")
        return

    rows=[json.loads(l) for l in qfile.open("r",encoding="utf-8")]
    rows=[r for r in rows if r.get("text","").strip()]

    texts=[r["text"].strip() for r in rows]
    ids=[r.get("id") or h(r["text"]) for r in rows]

    mat = embed_batch(texts)
    faiss.normalize_L2(mat)

    index=faiss.IndexFlatIP(mat.shape[1])
    id_index=faiss.IndexIDMap2(index)
    id_index.add_with_ids(mat, np.array([int(int(h(i)[0:8],16)) for i in ids], dtype="int64"))

    INDEX_DIR.mkdir(parents=True,exist_ok=True)
    faiss.write_index(id_index, str(INDEX_DIR/"index.faiss"))
    np.save(INDEX_DIR/"vectors.npy", mat)

    with (INDEX_DIR/"meta.jsonl").open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

    manifest={
      "set_id": set_id,
      "emb_model": EMB_MODEL,
      "dim": int(mat.shape[1]),
      "count": int(mat.shape[0]),
      "created_at": datetime.datetime.utcnow().isoformat()+"Z"
    }
    (INDEX_DIR/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] {set_id} index with {manifest['count']} vectors")

if args.set_id.upper()=="ALL":
    for p in sorted(Path("output").glob("SET*")):
        if p.is_dir(): build_for_set(p.name)
else:
    build_for_set(args.set_id)
