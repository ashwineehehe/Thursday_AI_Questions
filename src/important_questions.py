#!/usr/bin/env python3
# file: important_questions.py
import argparse, io, json, re, time, hashlib
from pathlib import Path

import fitz                   # PyMuPDF
from PIL import Image
import pytesseract
import ollama

#  helpers 
def h(txt: str) -> str:
    return hashlib.sha1(txt.strip().lower().encode("utf-8")).hexdigest()

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")

def read_units(units_arg: str | None, fallback_list: list[str]) -> list[str]:
    if units_arg:
        p = Path(units_arg)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return data
            except Exception:
                pass
    return fallback_list

def llm_chat(model: str, messages: list[dict], temperature: float = 0.0, retries: int = 3):
    for i in range(retries):
        try:
            return ollama.chat(model=model, messages=messages, options={"temperature": temperature})
        except Exception:
            time.sleep(0.7 * (i + 1))
    return {"message": {"content": ""}}

#  OCR 
def extract_text_with_ocr(pdf_path: Path, ocr_min_chars: int) -> str:
    doc = fitz.open(str(pdf_path))
    page_texts = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        if len(txt.strip()) < ocr_min_chars:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            txt += "\n" + pytesseract.image_to_string(img, config="--oem 3 --psm 6")
        page_texts.append(txt.replace("\u00A0", " ").strip())
    doc.close()
    return "\n".join(page_texts)

# LLM tasks 
def llm_extract_questions(raw_text: str, model: str) -> list[dict]:
    sys_msg = "Extract atomic exam questions. Output JSONL, one object per line with fields {id, text}. No extra text."
    user_msg = (
        "From the following text, extract ALL exam questions. "
        "If a question has parts (a)(b)(c), split into separate atomic questions.\n\n"
        + raw_text[:12000]
    )
    res = llm_chat(model, [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}], 0)
    qs = []
    for line in res["message"]["content"].splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("text"):
                qs.append({"id": obj.get("id") or h(obj["text"]), "text": obj["text"].strip()})
        except Exception:
            # ignore non-JSONL lines
            pass
    return qs

def llm_classify_chapter(qtext: str, units: list[str], model: str) -> str:
    prompt = (
        "Assign this exam question to exactly one of these units.\n\n"
        f"{json.dumps(units, ensure_ascii=False)}\n\n"
        "Question:\n"
        f"{qtext}\n\n"
        "Return ONLY the exact unit string from the list; if unsure, return Unknown."
    )
    res = llm_chat(model, [{"role": "user", "content": prompt}], 0)
    ans = res["message"]["content"].strip()
    for u in units:
        if u.lower() == ans.lower():
            return u
    # (lenient) also allow 'contains'
    for u in units:
        if u.lower() in ans.lower():
            return u
    return "Unknown"

def llm_is_important(qtext: str, model: str) -> bool:
    prompt = "Is this an IMPORTANT exam question based on common trends? Reply strictly 'Yes' or 'No'.\n\n" + qtext
    res = llm_chat(model, [{"role": "user", "content": prompt}], 0)
    return "yes" in res["message"]["content"].strip().lower()

# `main
def main():
    parser = argparse.ArgumentParser(description="Extract important questions per SET and write per-chapter JSONs.")
    parser.add_argument("--pdf", required=True, help="Path to the SET PDF (e.g., past papers/SET1.pdf)")
    parser.add_argument("--out-dir", required=True, help="Output dir for this SET (e.g., output/SET1)")
    parser.add_argument("--model", default="llama3", help="Ollama chat model")
    parser.add_argument("--units", default=None, help="Optional path to units.json (list of strings)")
    parser.add_argument("--ocr-min", type=int, default=120, help="Min plain text chars before using OCR")
    args = parser.parse_args()

    PDF_PATH = Path(args.pdf)
    OUT_DIR = Path(args.out_dir)
    LLM_MODEL = args.model
    OCR_MIN_CHARS = args.ocr_min

    # default fallback units (used if --units not provided or invalid)
    DEFAULT_UNITS = [
        "Unit 1.1 Networking and Telecommunication",
        "Unit 1.2 Ethical and Social Issues in ICT",
        "Unit 1.3 Computer Security",
        "Unit 1.4 E-commerce",
        "Unit 1.5 Contemporary Technology",
        "Unit 1.6 Number System",
        "Unit 2.1 Database Management System",
        "Unit 3.1 Programming in QBASIC",
        "Unit 3.2 Modular Programming",
        "Unit 3.3 File Handling in QBASIC",
        "Unit 4.1 Structured Programing in C",
    ]
    UNITS = read_units(args.units, DEFAULT_UNITS)

    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[parse] {PDF_PATH.name}")
    full_text = extract_text_with_ocr(PDF_PATH, OCR_MIN_CHARS)

    # 1) extract all atomic questions
    raw_qs = llm_extract_questions(full_text, LLM_MODEL)

    # 2) classify + importance vote (with simple caches to avoid repeat calls)
    cache_dir = OUT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cls_cache_path = cache_dir / "classify.json"
    imp_cache_path = cache_dir / "important.json"
    classify_cache = json.loads(cls_cache_path.read_text()) if cls_cache_path.exists() else {}
    important_cache = json.loads(imp_cache_path.read_text()) if imp_cache_path.exists() else {}

    final = []
    for i, q in enumerate(raw_qs, 1):
        t = (q.get("text") or "").strip()
        if not t:
            continue

        k = h(t)
        # chapter
        chap = classify_cache.get(k)
        if not chap:
            chap = llm_classify_chapter(t, UNITS, LLM_MODEL)
            classify_cache[k] = chap
            cls_cache_path.write_text(json.dumps(classify_cache, ensure_ascii=False), encoding="utf-8")

        # important?
        imp = important_cache.get(k)
        if imp is None:
            imp = llm_is_important(t, LLM_MODEL)
            important_cache[k] = imp
            imp_cache_path.write_text(json.dumps(important_cache, ensure_ascii=False), encoding="utf-8")

        if not imp:
            continue

        final.append({
            "id": f"{PDF_PATH.stem}::Q{i:04d}",
            "paper_id": PDF_PATH.stem,
            "text": t,
            "chapter": chap,
        })

    # 3) write per-SET files
    # 3a) questions.jsonl
    with (OUT_DIR / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in final:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # 3b) important_all_chapters.json
    by_chap: dict[str, list[dict]] = {}
    for q in final:
        by_chap.setdefault(q["chapter"], []).append(q)
    (OUT_DIR / "important_all_chapters.json").write_text(
        json.dumps(by_chap, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 3c) per-SET per-chapter JSONs
    per_set_chap_dir = OUT_DIR / "chapters"
    per_set_chap_dir.mkdir(parents=True, exist_ok=True)
    for chapter, rows in by_chap.items():
        (per_set_chap_dir / f"{slugify(chapter)}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 4) global chapter files (append + de-dup)
    global_dir = Path("./chapters")
    global_dir.mkdir(parents=True, exist_ok=True)
    for chapter, rows in by_chap.items():
        cpath = global_dir / f"{slugify(chapter)}.json"
        existing = []
        if cpath.exists():
            try:
                existing = json.loads(cpath.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        # de-dup by (paper_id, text)
        seen = {(r.get("paper_id"), r.get("text")) for r in existing}
        for r in rows:
            key = (r.get("paper_id"), r.get("text"))
            if key not in seen:
                existing.append(r)
                seen.add(key)
        cpath.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] Wrote {len(final)} important questions to {OUT_DIR}")
    print("[done] Per-chapter files updated under:")
    print(f"       - {per_set_chap_dir} (per-SET)")
    print(f"       - {global_dir} (global)")

if __name__ == "__main__":
    main()
