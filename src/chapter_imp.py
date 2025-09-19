#!/usr/bin/env python3
import json, re
from pathlib import Path
from difflib import SequenceMatcher

# ---------- HARD-CODED SETTINGS ----------
OUTPUT_ROOT = Path("./output")                 # scans output/SET1 .. output/SET36
CHAPTER     = "Unit 3.1 Programming in QBASIC" # <--- set the unit you want here
TOP_N       = 12
# -----------------------------------------

# -------- Strong post-filter patterns (per unit) --------
INCLUDE = {
    "Unit 1.1 Networking and Telecommunication": re.compile(
        r"\b(network|lan|wan|man|topology|star|bus|ring|mesh|protocol|ip\b|tcp|udp|http|https|dns|"
        r"mac address|bandwidth|isp|wi-?fi|wifi|router|switch|hub|ethernet|client|server|peer[- ]?to[- ]?peer|"
        r"modem|gateway|subnet|ip address|telecommunication|telephony|voip)\b", re.I),

    "Unit 1.2 Ethical and Social Issues in ICT": re.compile(
        r"\b(ethic|ethical|social issue|digital citizenship|digital footprint|privacy|cyber bullying|harassment|"
        r"copyright|plagiarism|fair use|netiquette|online safety|digital reputation)\b", re.I),

    "Unit 1.3 Computer Security": re.compile(
        r"\b(security|computer security|information security|confidentiality|integrity|availability|cia triad|"
        r"virus|antivirus|malware|spyware|ransomware|phishing|spoofing|firewall|encryption|decryption|"
        r"password policy|two[- ]?factor|2fa|backup|ups|social engineering)\b", re.I),

    "Unit 1.4 E-commerce": re.compile(
        r"\b(e-?commerce|m-?commerce|online shopping|payment gateway|transaction|shopping cart|"
        r"digital wallet|paypal|khalti|fonepay|esewa|upi|net banking|b2b|b2c|c2c|g2c|invoice|order|delivery|refund)\b", re.I),

    "Unit 1.5 Contemporary Technology": re.compile(
        r"\b(contemporary technology|ai|artificial intelligence|machine learning|ml\b|deep learning|dl\b|"
        r"iot|internet of things|cloud computing|ar\b|vr\b|virtual reality|augmented reality|blockchain|"
        r"big data|analytics|wearable|drones|3d printing)\b", re.I),

    "Unit 1.6 Number System": re.compile(
        r"(\([01]+\)₂|\([0-7]+\)₈|\([0-9A-Fa-f]+\)₁₆|base\s*(2|8|10|16)|\b(binary|octal|decimal|hexadecimal|"
        r"conversion|convert|complement|bit|nibble|byte|ascii|bcd)\b)", re.I),

    "Unit 2.1 Database Management System": re.compile(
        r"\b(dbms|database|table|field|record|tuple|relation|primary key|foreign key|candidate key|"
        r"query|form|report|ms-?access|data type|memo|ole|criteria|sorting|filter|sql|select|update|insert|delete)\b", re.I),

    "Unit 3.1 Programming in QBASIC": re.compile(
        r"\b(qbasic|q-basic|declare\s+sub|declare\s+function|sub\b|function\b|while\s+wend|for\s+.*\s+next|"
        r"do\s+.*\s+loop|if\s+.*\s+then|elseif|else|end if|input\s?#|print\s?#|open\s+\"|close\s+#|"
        r"line\s+input|write\s+#|put\s+#|get\s+#|eof\(|"
        r"mid\$|left\$|right\$|len\$|ucase\$|lcase\$|instr|val|int\b|sqr\b|rnd\b|"
        r"files\b|kill\b|name\b|shell\b)\b", re.I),

    "Unit 3.2 Modular Programming": re.compile(
        r"\b(modular programming|module|subroutine|procedure|sub\b|function\b|parameters|arguments|scope|"
        r"local variable|global variable|interface|reuse|reusability|abstraction)\b", re.I),

    "Unit 3.3 File Handling in QBASIC": re.compile(
        r"\b(file handling|sequential file|random file|binary file|open\s+\"|close\s+#|input\s?#|print\s?#|"
        r"write\s+#|put\s+#|get\s+#|eof\(|field\b|len\b|append\s+as\s+#)\b", re.I),

    "Unit 4.1 Structured Programing in C": re.compile(
        r"(#include|\bstdio\.h\b|\bconio\.h\b|\bctype\.h\b|\bstring\.h\b|\bmath\.h\b|"
        r"\bint\s+main\b|\bvoid\s+main\b|\bscanf\s*\(|\bprintf\s*\(|\bgets\s*\(|\bputs\s*\(|"
        r"\bchar\b|\bint\b|\bfloat\b|\bdouble\b|\bif\s*\(|\belse\b|\bswitch\s*\(|\bcase\b|\bfor\s*\(|\bwhile\s*\(|\bdo\s*\{|"
        r"\breturn\b|\bsizeof\b)", re.I),
}

EXCLUDE = {
    # Each unit excludes obvious signals from other units to prevent bleed-through.
    "Unit 1.1 Networking and Telecommunication": re.compile(
        r"(qbasic|declare\s+sub|#include|\bscanf\s*\(|\bprintf\s*\(|primary key|ms-?access)", re.I),
    "Unit 1.2 Ethical and Social Issues in ICT": re.compile(
        r"(qbasic|#include|\bscanf\s*\(|\bprintf\s*\(|primary key|router|switch)", re.I),
    "Unit 1.3 Computer Security": re.compile(
        r"(qbasic|#include|\bscanf\s*\(|\bprintf\s*\(|primary key|ms-?access|topology)", re.I),
    "Unit 1.4 E-commerce": re.compile(
        r"(qbasic|#include|\bscanf\s*\(|\bprintf\s*\(|primary key|ms-?access|router|switch|jpeg|png)", re.I),
    "Unit 1.5 Contemporary Technology": re.compile(
        r"(qbasic|#include|\bscanf\s*\(|\bprintf\s*\(|primary key|sql|router|switch)", re.I),
    "Unit 1.6 Number System": re.compile(
        r"(qbasic|#include|\bscanf\s*\(|\bprintf\s*\(|database|primary key|router|switch|browser)", re.I),
    "Unit 2.1 Database Management System": re.compile(
        r"(qbasic|#include|\bscanf\s*\(|\bprintf\s*\(|router|switch|http|https|jpeg|png|animation)", re.I),
    "Unit 3.1 Programming in QBASIC": re.compile(
        r"(#include|\bscanf\s*\(|\bprintf\s*\(|\bint\s+main\b|\bvoid\s+main\b|in\s+C( language)?\b)", re.I),
    "Unit 3.2 Modular Programming": re.compile(
        r"(#include|\bscanf\s*\(|\bprintf\s*\(|router|switch|primary key|ms-?access)", re.I),
    "Unit 3.3 File Handling in QBASIC": re.compile(
        r"(#include|\bscanf\s*\(|\bprintf\s*\(|router|switch|primary key|ms-?access)", re.I),
    "Unit 4.1 Structured Programing in C": re.compile(
        r"\b(qbasic|q-basic|declare\s+sub|declare\s+function|while\s+wend|for\s+.*\s+next|"
        r"do\s+.*\s+loop|mid\$|len\$|ucase\$|lcase\$|files\b|kill\b|name\b|open\s+\"|close\s+#)\b", re.I),
}

def chapter_filter(text: str, chapter: str) -> bool:
    t = text or ""
    inc = INCLUDE.get(chapter)
    exc = EXCLUDE.get(chapter)
    if inc and not inc.search(t):
        return False
    if exc and exc.search(t):
        return False
    return True

def collect_from_all_sets(out_root: Path, chapter: str):
    items = []
    for set_dir in sorted(out_root.glob("SET*")):
        f = set_dir / "important_all_chapters.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if chapter not in data:
            continue
        for q in data[chapter]:
            q2 = dict(q)
            q2["_set"] = set_dir.name
            items.append(q2)
    return items

def dedup_by_text(items, sim=0.90):
    keep = []
    for it in items:
        t = (it.get("text") or "").strip()
        if not t:
            continue
        placed = False
        tl = t.lower()
        for k in keep:
            if SequenceMatcher(None, tl, (k["text"] or "").lower()).ratio() >= sim:
                k["score"] = max(k.get("score", 0.0), it.get("score", 0.0))
                k["papers_count"] = max(k.get("papers_count", 0), it.get("papers_count", 0))
                k["_sets"].add(it.get("_set", "?"))
                placed = True
                break
        if not placed:
            it["_sets"] = {it.get("_set", "?")}
            keep.append(it)
    return keep

def main():
    all_items = collect_from_all_sets(OUTPUT_ROOT, CHAPTER)
    if not all_items:
        print(f"No items found for chapter '{CHAPTER}' under {OUTPUT_ROOT.resolve()}")
        return

    filtered = [q for q in all_items if chapter_filter(q.get("text",""), CHAPTER)]
    if not filtered:
        print(f"No items matched post-filter for '{CHAPTER}'. You may loosen patterns.")
        return

    canon = dedup_by_text(filtered, sim=0.90)
    canon.sort(key=lambda x: (-x.get("score", 0.0), -x.get("papers_count", 0), x.get("text", "")))

    topk = canon[:TOP_N]
    print(f"\nImportant Questions for: {CHAPTER}")
    print(f"(showing {len(topk)} of {len(canon)} after post-filter & dedup; from {len(all_items)} raw across SETs)\n")
    for i, q in enumerate(topk, start=1):
        sets = ", ".join(sorted(list(q.get("_sets", []))))
        print(f"{i}. {q.get('text','').strip()}")
        print(f"   — sets=[{sets}]  score={q.get('score')}  seen_in={q.get('papers_count')} paper(s)\n")

if __name__ == "__main__":
    main()
