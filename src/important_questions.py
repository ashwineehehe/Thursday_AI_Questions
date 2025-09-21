#!/usr/bin/env python3
import os, re, json, io, argparse
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher

# ---- Optional Ollama (kept OFF by default to avoid RAM issues) ----
USE_OLLAMA = False           # set True to use Ollama for importance/chapter (needs llama3 or tinyllama)
OLLAMA_MODEL = "tinyllama"   # lighter than llama3; change if you have RAM
if USE_OLLAMA:
    import ollama

# ---- IO CONFIG ----
# You can pass --pdf and --out on the CLI; these are defaults:
PDF_PATH= Path(r"past papers/SET1.pdf")
OUT_DIR = Path("./output/SET1")
TOP_N = 12

# ---- Chapter hints (regex) ----
CHAPTER_HINTS = [
    ("Unit 1.1 Networking and Telecommunication",
     r"\b(network|lan|wan|man|topology|star|bus|ring|mesh|protocol|ip\b|tcp|udp|http|https|dns|"
     r"mac address|bandwidth|isp|wi-?fi|wifi|router|switch|hub|ethernet|client|server|peer[- ]?to[- ]?peer|"
     r"modem|gateway|subnet|ip address|telecommunication|telephony|voip)\b"),
    ("Unit 1.2 Ethical and Social Issues in ICT",
     r"\b(ethic|ethical|social issue|digital citizenship|digital footprint|privacy|cyber bullying|harassment|"
     r"copyright|plagiarism|fair use|netiquette|online safety|digital reputation)\b"),
    ("Unit 1.3 Computer Security",
     r"\b(security|computer security|information security|confidentiality|integrity|availability|cia triad|"
     r"virus|antivirus|malware|spyware|ransomware|phishing|spoofing|firewall|encryption|decryption|"
     r"password policy|two[- ]?factor|2fa|backup|ups|social engineering)\b"),
    ("Unit 1.4 E-commerce",
     r"\b(e-?commerce|m-?commerce|online shopping|payment gateway|transaction|shopping cart|"
     r"digital wallet|paypal|khalti|fonepay|esewa|upi|net banking|b2b|b2c|c2c|g2c|invoice|order|delivery|refund)\b"),
    ("Unit 1.5 Contemporary Technology",
     r"\b(contemporary technology|ai|artificial intelligence|machine learning|ml\b|deep learning|dl\b|"
     r"iot|internet of things|cloud computing|ar\b|vr\b|virtual reality|augmented reality|blockchain|"
     r"big data|analytics|wearable|drones|3d printing)\b"),
    ("Unit 1.6 Number System",
     r"(\([01]+\)₂|\([0-7]+\)₈|\([0-9A-Fa-f]+\)₁₆|base\s*(2|8|10|16)|\b(binary|octal|decimal|hexadecimal|"
     r"conversion|convert|complement|bit|nibble|byte|ascii|bcd)\b)"),
    ("Unit 2.1 Database Management System",
     r"\b(dbms|database|table|field|record|tuple|relation|primary key|foreign key|candidate key|"
     r"query|form|report|ms-?access|data type|memo|ole|criteria|sorting|filter|sql|select|update|insert|delete)\b"),
    ("Unit 3.1 Programming in QBASIC",
     r"\b(qbasic|q-basic|declare\s+sub|declare\s+function|sub\b|function\b|while\s+wend|for\s+.*\s+next|"
     r"do\s+.*\s+loop|if\s+.*\s+then|elseif|else|end if|input\s?#|print\s?#|open\s+\"|close\s+#|"
     r"line\s+input|write\s+#|put\s+#|get\s+#|eof\(|"
     r"mid\$|left\$|right\$|len\$|ucase\$|lcase\$|instr|val|int\b|sqr\b|rnd\b|"
     r"files\b|kill\b|name\b|shell\b)\b"),
    ("Unit 3.2 Modular Programming",
     r"\b(modular programming|module|subroutine|procedure|sub\b|function\b|parameters|arguments|scope|"
     r"local variable|global variable|interface|reuse|reusability|abstraction)\b"),
    ("Unit 3.3 File Handling in QBASIC",
     r"\b(file handling|sequential file|random file|binary file|open\s+\"|close\s+#|input\s?#|print\s?#|"
     r"write\s+#|put\s+#|get\s+#|eof\(|field\b|len\b|append\s+as\s+#)\b"),
    ("Unit 4.1 Structured Programing in C",
     r"(#include|\bstdio\.h\b|\bconio\.h\b|\bctype\.h\b|\bstring\.h\b|\bmath\.h\b|"
     r"\bint\s+main\b|\bvoid\s+main\b|\bscanf\s*\(|\bprintf\s*\(|\bgets\s*\(|\bputs\s*\(|"
     r"\bchar\b|\bint\b|\bfloat\b|\bdouble\b|\bif\s*\(|\belse\b|\bswitch\s*\(|\bcase\b|\bfor\s*\(|\bwhile\s*\(|\bdo\s*\{|"
     r"\breturn\b|\bsizeof\b)")
]

# ---- Section anchors + patterns ----
BLOCK_TITLES = re.compile(
    r'(answer the following|choose|select the correct answer|match the following|write full form|'
    r'full forms|give appropriate technical terms|write appropriate technical term|'
    r'convert\/?calculate|re-?write|write the output|study the following program|'
    r'debug the given program|programming questions?)', re.I)
QNUM   = re.compile(r'^\s*\d+[\.\)]\s+', re.M)
SUB    = re.compile(r'^\s*[a-e]\)\s+', re.M)
ROMAN  = re.compile(r'^\s*\([ivx]+\)\s+', re.M|re.I)
BULLET = re.compile(r'^\s*[•\-\u2022]\s+', re.M)
INLINE_SUB = re.compile(r'(?<!\w)\(([a-z])\)\s+', re.I)
LEAD_NUM_RE = re.compile(r'^\s*(?:q?\d+[\.\)]|\(\d+\)|\d+\)|[a-e]\)|\([ivx]+\)|[•\-\u2022])\s+', re.M)
GENERIC_PAT = re.compile(
    r'^(answer the following|write the full form|write appropriate technical term|match the following|choose|select the correct answer|state whether|convert\/?calculate|programming questions?|group\s+[abc])',
    re.I)
ONLY_MATHY = re.compile(r'^[\s\(\)\[\]₀-₉0-9xX÷\+\-\*=\?\.]+$')
MIN_WORDS = 5

FMT_W = {
    "PROGRAMMING": 0.6, "PROGRAM_DEBUG": 0.6, "PROGRAM_OUTPUT": 0.55, "PROGRAM_READ": 0.55,
    "CONVERSION": 0.5, "QA": 0.4, "MCQ": 0.2, "MATCH": 0.15, "FULL_FORM": 0.1, "TECH_TERM": 0.1, "GENERAL": 0.25
}

def fmt_from_heading(heading: str) -> str:
    h = heading.lower()
    if "choose" in h or "select the correct answer" in h: return "MCQ"
    if "match the following" in h: return "MATCH"
    if "full form" in h: return "FULL_FORM"
    if "technical term" in h: return "TECH_TERM"
    if "convert" in h or "calculate" in h: return "CONVERSION"
    if "write the output" in h: return "PROGRAM_OUTPUT"
    if "debug" in h or "re-write" in h or "rewrite" in h: return "PROGRAM_DEBUG"
    if "study the following program" in h: return "PROGRAM_READ"
    if "programming" in h: return "PROGRAMMING"
    if "answer the following" in h: return "QA"
    return "GENERAL"

def norm_text(s: str) -> str:
    s2 = s.replace("\u00A0"," ").replace("“","\"").replace("”","\"").replace("’","'").replace("‘","'")
    s2 = re.sub(r'\s+', ' ', s2).strip().lower()
    s2 = LEAD_NUM_RE.sub('', s2).strip()
    return s2

def is_keep_question(text: str) -> bool:
    t = text.strip()
    if GENERIC_PAT.match(t): return False
    if ONLY_MATHY.match(t): return False
    if len(t.split()) < MIN_WORDS: return False
    return True

def guess_chapter_regex(text: str) -> str:
    t = text.lower()
    for chap, pat in CHAPTER_HINTS:
        if re.search(pat, t):
            return chap
    return "Unit 3.1 Programming in QBASIC" if "qbasic" in t else "Unit 4.1 Structured Programing in C" if "#include" in t or "scanf(" in t else "Unit 2.1 Database Management System" if "primary key" in t or "ms-access" in t else "Unit 1.3 Computer Security"

def ask_ollama_yesno(msg: str) -> bool:
    if not USE_OLLAMA:
        return True
    r = ollama.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":msg}])
    return "yes" in r["message"]["content"].lower()

def ask_ollama_chapter(question_text: str) -> str:
    if not USE_OLLAMA:
        return guess_chapter_regex(question_text)
    opts = ", ".join([c for c,_ in CHAPTER_HINTS])
    prompt = f"Pick one exact chapter from these options:\n{opts}\n\nQuestion:\n{question_text}\n\nAnswer with the exact chapter string only."
    r = ollama.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":prompt}])
    out = r["message"]["content"].strip()
    return out if any(out.startswith(c.split()[0]) for c,_ in CHAPTER_HINTS) else guess_chapter_regex(question_text)

# -------- PDF + OCR ----------
def extract_text_with_ocr(pdf_path: Path, ocr_min_chars: int = 120):
    import fitz
    from PIL import Image
    import pytesseract
    doc = fitz.open(str(pdf_path))
    page_texts = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = (page.get_text("text") or "").strip()
        if len(txt) < ocr_min_chars:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            txt = (txt + "\n" + (pytesseract.image_to_string(img, config="--oem 3 --psm 6") or "")).strip()
        try:
            for xref, *_ in page.get_images(full=True):
                pm = fitz.Pixmap(doc, xref)
                if pm.width >= 200 and pm.height >= 200:
                    img = Image.open(io.BytesIO(pm.tobytes("png")))
                    o = pytesseract.image_to_string(img, config="--oem 3 --psm 6") or ""
                    if len(o.strip()) > 10:
                        txt += "\n" + o.strip()
        except Exception:
            pass
        txt = txt.replace("\u00A0"," ")
        txt = re.sub(r'[ \t]+', ' ', txt)
        page_texts.append(txt)
    doc.close()
    return "\n".join(page_texts)

# -------- splitting ----------
def split_blocks(text: str):
    matches = list(BLOCK_TITLES.finditer(text))
    if not matches:
        return [("GENERAL", text)]
    parts, last, last_heading = [], 0, "GENERAL"
    for m in matches:
        if m.start() > last:
            parts.append((last_heading, text[last:m.start()]))
        last_heading, last = m.group(0), m.start()
    parts.append((last_heading, text[last:]))
    return parts

def split_questions_in_block(block_text: str):
    starts = []
    for rgx in (QNUM, SUB, ROMAN, BULLET):
        for m in rgx.finditer(block_text):
            starts.append(m.start())
    starts = sorted(set(starts))
    items = []
    if not starts:
        t = block_text.strip()
        if len(t.split()) >= 6:
            items.append(t)
        return items
    starts.append(len(block_text))
    for i in range(len(starts)-1):
        chunk = block_text[starts[i]:starts[i+1]].strip()
        if not chunk: continue
        parts = INLINE_SUB.split(chunk)
        if len(parts) > 1:
            stem = parts[0].strip().rstrip(':')
            for j in range(1, len(parts), 2):
                tag = parts[j]; body = parts[j+1].strip()
                if body: items.append(f"{stem} ({tag}). {body}")
        else:
            items.append(chunk)
    cleaned = []
    for it in items:
        s = LEAD_NUM_RE.sub('', it).strip()
        s = re.sub(r'\s+', ' ', s)
        if len(s.split()) >= 4:
            cleaned.append(s)
    return cleaned

def is_near_dup(a_norm: str, b_norm: str) -> bool:
    if a_norm == b_norm:
        return True
    if len(a_norm.split()) < 6 or len(b_norm.split()) < 6:
        return False
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= 0.90

# -------- pipeline ----------
def process_pdf(pdf_path: Path):
    text = extract_text_with_ocr(pdf_path)
    blocks = split_blocks(text)
    paper_id = pdf_path.stem
    questions, seen_norm, qid = [], set(), 0

    for heading, block in blocks:
        fmt = fmt_from_heading(heading)
        for txt in split_questions_in_block(block):
            if not is_keep_question(txt):
                continue
            norm = norm_text(txt)
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            qid += 1
            chapter = ask_ollama_chapter(txt)
            important = ask_ollama_yesno(f"Is this an IMPORTANT exam question? Answer yes/no.\n\n{txt}")
            questions.append({
                "id": f"{paper_id}::Q{qid:04d}",
                "paper_id": paper_id,
                "section_heading": heading.strip(),
                "format": fmt,
                "text": txt.strip(),
                "norm": norm,
                "chapter": chapter,
                "important": bool(important),
            })
    return questions

def rank_important(all_questions, top_n=TOP_N, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_chap = defaultdict(list)
    for q in all_questions:
        by_chap[q["chapter"]].append(q)

    per_chapter_top = {}
    for chap, items in by_chap.items():
        canon = []
        for it in items:
            if not it.get("important", True):   # include all if not using Ollama
                continue
            placed = False
            for c in canon:
                if is_near_dup(it["norm"], c["norm"]):
                    c["papers"].add(it["paper_id"])
                    if len(it["text"]) > len(c["text"]):
                        c["text"] = it["text"]; c["format"] = it["format"]; c["section_heading"] = it["section_heading"]
                    placed = True
                    break
            if not placed:
                canon.append({
                    "text": it["text"], "norm": it["norm"],
                    "format": it["format"], "section_heading": it["section_heading"],
                    "papers": set([it["paper_id"]]),
                })
        scored = []
        for c in canon:
            rec = len(c["papers"])
            score = rec + FMT_W.get(c["format"], 0.25)
            scored.append({
                "text": c["text"], "format": c["format"], "section_heading": c["section_heading"],
                "papers_count": rec, "paper_ids": sorted(list(c["papers"])), "score": round(score, 3),
            })
        scored.sort(key=lambda x: (-x["score"], -x["papers_count"], x["format"], x["text"]))
        per_chapter_top[chap] = scored[:top_n]

    # write files
    (out_dir / "important_all_chapters.json").write_text(
        json.dumps(per_chapter_top, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return per_chapter_top

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=PDF_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf.resolve()}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[parse] {args.pdf.name}")

    qs = process_pdf(args.pdf)
    print(f"[debug] extracted raw: {len(qs)}")
    print("[debug] by chapter:", Counter(q["chapter"] for q in qs))

    # dump all extracted for embedding
    with (args.out / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[done] wrote: {args.out / 'questions.jsonl'}")

    rank_important(qs, top_n=TOP_N, out_dir=args.out)
    print(f"[done] wrote: {args.out / 'important_all_chapters.json'}")

if __name__ == "__main__":
    main()
