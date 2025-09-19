#!/usr/bin/env python3
import os, re, json, io
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import fitz                     # PDF parsing (PyMuPDF)
from PIL import Image
import pytesseract              # OCR
import ollama

# CONFIG 
PDF_PATH = Path(r"past papers/SET1.pdf")  # <-- ensure exact filename
OUT_DIR = Path("./output/SET1")                    # <- where JSON will be written
TOP_N = 12                                # <- top important questions per chapter

# Use your exact book chapter names here (tweak regex as needed)
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

# Section anchors + question-start patterns
BLOCK_TITLES = re.compile(
    r'(answer the following|choose|select the correct answer|match the following|write full form|'
    r'full forms|give appropriate technical terms|write appropriate technical term|'
    r'convert\/?calculate|re-?write|write the output|study the following program|'
    r'debug the given program|programming questions?)',
    re.I
)
QNUM   = re.compile(r'^\s*\d+[\.\)]\s+', re.M)
SUB    = re.compile(r'^\s*[a-e]\)\s+', re.M)
ROMAN  = re.compile(r'^\s*\([ivx]+\)\s+', re.M|re.I)
BULLET = re.compile(r'^\s*[•\-\u2022]\s+', re.M)
INLINE_SUB = re.compile(r'(?<!\w)\(([a-z])\)\s+', re.I)
LEAD_NUM_RE = re.compile(r'^\s*(?:q?\d+[\.\)]|\(\d+\)|\d+\)|[a-e]\)|\([ivx]+\)|[•\-\u2022])\s+', re.M)
GENERIC_PAT = re.compile(
    r'^(answer the following|write the full form|write appropriate technical term|match the following|choose|select the correct answer|state whether|convert\/?calculate|programming questions?|group\s+[abc])',
    re.I
)
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

def ask_ollama_for_chapter(question_text: str) -> str:
    prompt = f"Which chapter does this question belong to? Options: {', '.join([c for c,_ in CHAPTER_HINTS])}. Question: {question_text}"
    resp = ollama.chat(model="llama3", messages=[{"role":"user","content":prompt}])
    return resp["message"]["content"].strip()

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

def is_important(question_text: str) -> bool:
    prompt = f"Given the exam trend, is this an IMPORTANT question? Answer only Yes or No.\n\n{question_text}"
    resp = ollama.chat(model="llama3", messages=[{"role":"user","content":prompt}])
    return "yes" in resp["message"]["content"].lower()

# PDF + OCR
def extract_text_with_ocr(pdf_path: Path, ocr_min_chars: int = 120):
    """
    Per page: try PyMuPDF text. If too short, rasterize page and OCR with pytesseract.
    Also OCR large embedded images.
    Returns a single joined text string.
    """
    import fitz
    from PIL import Image
    import pytesseract

    doc = fitz.open(str(pdf_path))
    page_texts = []

    for i in range(doc.page_count):
        page = doc.load_page(i)

        # 1) native text
        txt = page.get_text("text") or ""
        txt = txt.strip()
        got_text = len(txt) >= ocr_min_chars

        if not got_text:
            # 2) OCR the page rendered as image (300dpi)
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_page = pytesseract.image_to_string(img, config="--oem 3 --psm 6") or ""
            txt = (txt + "\n" + ocr_page).strip()

        # 3) OCR large embedded images too (sometimes tables/diagrams contain text)
        try:
            for xref, *_ in page.get_images(full=True):
                pix = fitz.Pixmap(doc, xref)
                if pix.width < 200 or pix.height < 200:
                    continue
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_img = pytesseract.image_to_string(img, config="--oem 3 --psm 6") or ""
                if len(ocr_img.strip()) > 10:
                    txt += "\n" + ocr_img.strip()
        except Exception:
            pass

        # light cleanup
        txt = txt.replace("\u00A0"," ")
        txt = re.sub(r'[ \t]+', ' ', txt)
        page_texts.append(txt)

    doc.close()
    return "\n".join(page_texts)

# Splitting 
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

# Per-PDF pipeline 
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
            questions.append({
                "id": f"{paper_id}::Q{qid:04d}",
                "paper_id": paper_id,
                "section_heading": heading.strip(),
                "format": fmt,
                "text": txt.strip(),
                "norm": norm,
                "chapter": guess_chapter(txt),
            })
    return questions

# Rank & Write 
def rank_important(all_questions, top_n=TOP_N, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_chap = defaultdict(list)
    for q in all_questions:
        by_chap[q["chapter"]].append(q)

    per_chapter_top = {}
    for chap, items in by_chap.items():
        canon = []
        for it in items:
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

        safe = chap.lower().replace(" ", "_").replace("/", "_")
        with open(out_dir / f"important_{safe}.json", "w", encoding="utf-8") as f:
            json.dump({"chapter": chap, "top_questions": per_chapter_top[chap]}, f, indent=2, ensure_ascii=False)

    with open(out_dir / "important_all_chapters.json", "w", encoding="utf-8") as f:
        json.dump(per_chapter_top, f, indent=2, ensure_ascii=False)
    return per_chapter_top

# Main 
def main():
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[parse] {PDF_PATH.name}")
    try:
        qs = process_pdf(PDF_PATH)
        print(f"  -> {len(qs)} atomic questions")

        # rank top-N per chapter
        rank_important(qs, top_n=TOP_N, out_dir=OUT_DIR)

        # NEW: dump all extracted questions into JSONL for embedding pipeline
        questions_out = OUT_DIR / "questions.jsonl"
        with open(questions_out, "w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"[done] Wrote {questions_out}")

    except Exception as e:
        print(f"  !! failed: {e}")

    print(f"[done] Output location: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
