#!/usr/bin/env python3
import json, re, argparse
from pathlib import Path
from difflib import SequenceMatcher

OUTPUT_ROOT = Path("./output")   # scans output/SET1 .. SET36
TOP_N       = 12

# Map aliases to textbook units (so “QBASIC Programming” still maps correctly)
ALIASES = {
    "qbasic programming": "Unit 3.1 Programming in QBASIC",
    "programming in qbasic": "Unit 3.1 Programming in QBASIC",
    "modular programming": "Unit 3.2 Modular Programming",
    "file handling in qbasic": "Unit 3.3 File Handling in QBASIC",
    "structured programming in c": "Unit 4.1 Structured Programing in C",
    "database management system": "Unit 2.1 Database Management System",
    "number system": "Unit 1.6 Number System",
    "networking and telecommunication": "Unit 1.1 Networking and Telecommunication",
    "ethical and social issues in ict": "Unit 1.2 Ethical and Social Issues in ICT",
    "computer security": "Unit 1.3 Computer Security",
    "e-commerce": "Unit 1.4 E-commerce",
    "contemporary technology": "Unit 1.5 Contemporary Technology",
}

def normalize_chapter_name(s: str) -> str:
    key = s.strip().lower()
    return ALIASES.get(key, s.strip())

# Strong post-filter patterns (adapted from your earlier version)
INCLUDE = {
    "Unit 3.1 Programming in QBASIC": re.compile(
        r"\b(qbasic|q-basic|declare\s+sub|declare\s+function|sub\b|function\b|while\s+wend|for\s+.*\s+next|"
        r"do\s+.*\s+loop|if\s+.*\s+then|elseif|else|end if|input\s?#|print\s?#|open\s+\"|close\s+#|"
        r"line\s+input|write\s+#|put\s+#|get\s+#|eof\(|mid\$|left\$|right\$|len\$|ucase\$|lcase\$|instr|val|int\b|sqr\b|rnd\b|"
        r"files\b|kill\b|name\b|shell\b)\b",
        re.I
    ),
    "Unit 4.1 Structured Programing in C": re.compile(
        r"(#include|\bstdio\.h\b|\bconio\.h\b|\bscanf\s*\(|\bprintf\s*\(|\bint\s+main\b|\bvoid\s+main\b|\bchar\b|\bint\b|\bfloat\b|\bswitch\b)",
        re.I
    ),
    "Unit 2.1 Database Management System": re.compile(
        r"\b(dbms|database|table|field|record|primary key|foreign key|query|form|report|ms-?access|sql)\b", re.I
    ),
    "Unit 1.6 Number System": re.compile(
        r"(binary|octal|decimal|hexadecimal|base\s*(2|8|10|16)|\([01]+\)₂|\([0-7]+\)₈|\([0-9A-Fa-f]+\)₁₆|conversion|complement)", re.I
    ),
    "Unit 1.1 Networking and Telecommunication": re.compile(
        r"\b(network|topology|router|switch|hub|lan|wan|ip address|protocol|http|dns|modem|wifi|ethernet)\b", re.I
    ),
    "Unit 1.2 Ethical and Social Issues in ICT": re.compile(
        r"\b(ethic|digital citizenship|privacy|plagiarism|copyright|cyber bullying|netiquette)\b", re.I
    ),
    "Unit 1.3 Computer Security": re.compile(
        r"\b(virus|antivirus|malware|phishing|firewall|encryption|decryption|ransomware|backup|password)\b", re.I
    ),
    "Unit 1.4 E-commerce": re.compile(
        r"\b(e-?commerce|payment|transaction|shopping|wallet|khalti|esewa|fonepay|b2b|b2c)\b", re.I
    ),
    "Unit 1.5 Contemporary Technology": re.compile(
        r"\b(ai|machine learning|iot|cloud|ar|vr|blockchain|big data)\b", re.I
    ),
    "Unit 3.2 Modular Programming": re.compile(
        r"\b(module|modular|procedure|subroutine|parameter|scope|abstraction|reusability)\b", re.I
    ),
    "Unit 3.3 File Handling in QBASIC": re.compile(
        r"\b(sequential file|random file|open\s+\"|close\s+#|input\s?#|print\s?#|write\s+#|put\s+#|get\s+#|eof\(|field\b|append\s+as\s+#)\b", re.I
    ),
}

EXCLUDE = {
    "Unit 3.1 Programming in QBASIC": re.compile(r"(#include|\bscanf\s*\(|\bprintf\s*\()", re.I),
    "Unit 4.1 Structured Programing in C": re.compile(r"\b(qbasic|declare\s+sub|declare\s+function|while\s+wend)\b", re.I),
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
    wanted = normalize_chapter_name(chapter)
    items = []
    for set_dir in sorted(out_root.glob("SET*")):
        f = set_dir / "important_all_chapters.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        # debug what chapters this set has
        print(f"[debug] {set_dir.name} chapters ->", list(data.keys()))

        # exact or alias match
        pool = None
        if wanted in data:
            pool = data[wanted]
        else:
            # loose fallback: try alias keys by lowercase
            for k in data.keys():
                if normalize_chapter_name(k).lower() == wanted.lower():
                    pool = data[k]; break
        if not pool:
            continue

        for q in pool:
            q2 = dict(q); q2["_set"] = set_dir.name
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", required=True, help="Exact textbook unit name (aliases handled)")
    ap.add_argument("--out-root", default=str(OUTPUT_ROOT), help="Folder that contains SET* outputs")
    ap.add_argument("--top", type=int, default=TOP_N)
    args = ap.parse_args()

    chapter = normalize_chapter_name(args.chapter)
    items = collect_from_all_sets(Path(args.out_root), chapter)
    if not items:
        print(f"No items found for chapter '{chapter}' under {args.out_root}")
        return

    filtered = [q for q in items if chapter_filter(q.get("text",""), chapter)]
    if not filtered:
        print(f"No items matched post-filter for '{chapter}'. Loosen patterns.")
        return

    canon = dedup_by_text(filtered, sim=0.90)
    canon.sort(key=lambda x: (-x.get("score", 0.0), -x.get("papers_count", 0), x.get("text","")))
    topk = canon[:args.top]

    print(f"\nImportant Questions for chapter: {chapter}")
    print(f"(showing {len(topk)} of {len(canon)} after post-filter & dedup; from {len(items)} raw across SETs)\n")
    for i, q in enumerate(topk, start=1):
        sets = ", ".join(sorted(list(q.get("_sets", []))))
        print(f"{i}. {q.get('text','').strip()}")
        print(f"   — sets=[{sets}]  score={q.get('score')}  seen_in={q.get('papers_count')} paper(s)\n")

if __name__ == "__main__":
    main()
