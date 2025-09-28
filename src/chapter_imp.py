#!/usr/bin/env python3
# file: chapter_imp.py  (enhanced)
import json, re, numpy as np, faiss
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--chapter", required=True)
parser.add_argument("--top-n", type=int, default=20)
parser.add_argument("--write", action="store_true")
parser.add_argument("--dedup", action="store_true")
args = parser.parse_args()

OUTPUT_ROOT = Path("./output")
GLOBAL_OUT  = Path("./chapters")
GLOBAL_OUT.mkdir(parents=True, exist_ok=True)

def slugify(s):
    s = re.sub(r"[^a-z0-9]+","-",s.strip().lower()); return re.sub(r"-+","-",s).strip("-")

# 1) Collect
items=[]
for sdir in sorted(OUTPUT_ROOT.glob("SET*")):
    f=sdir/"important_all_chapters.json"
    if not f.exists(): continue
    data=json.loads(f.read_text(encoding="utf-8"))
    for q in data.get(args.chapter, []):
        q["_set"]=sdir.name
        items.append(q)

if not items:
    print(f"No questions found for {args.chapter}")
    raise SystemExit(0)

# 2) Optional dedup via text key (fast path)
def tkey(t): return re.sub(r"\s+"," ",t.strip().lower())
unique={}
for q in items:
    unique.setdefault(tkey(q["text"]), q)
items=list(unique.values())

# (Optional) vector-based dedup can be added if you load vectors from each set.

# 3) Trend score
from collections import defaultdict
by_text=defaultdict(lambda: {"sets": set(), "mentions": 0, "row": None})
for q in items:
    k=tkey(q["text"])
    by_text[k]["sets"].add(q["_set"])
    by_text[k]["mentions"]+=1
    by_text[k]["row"]=q

scored=[]
for k,v in by_text.items():
    score = 2.0*len(v["sets"]) + 0.5*v["mentions"]
    r = v["row"]
    scored.append({
        "text": r["text"],
        "paper_id": r.get("paper_id"),
        "sets": sorted(list(v["sets"])),
        "mentions": v["mentions"],
        "score": score
    })

scored.sort(key=lambda x:x["score"], reverse=True)
scored = scored[:args.top_n]

# 4) Output
if args.write:
    (GLOBAL_OUT/f"{slugify(args.chapter)}.json").write_text(
        json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8"
    )
else:
    print(f"\nImportant Questions for {args.chapter}\n")
    for i,q in enumerate(scored,1):
        print(f"{i}. {q['text']}  [sets:{len(q['sets'])}, score:{q['score']:.1f}]")
