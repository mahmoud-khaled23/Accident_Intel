#!/usr/bin/env python3
"""
clean_vtl.py - clean and re-chunk the NY Vehicle & Traffic Law (Title 7) JSONL
for use in a RAG system.

Usage:
    python clean_vtl.py title_7.jsonl out_dir [--max-chars 3500] [--min-chars 150]

Outputs (in out_dir):
    title_7_clean.jsonl     retrieval chunks (embed `embed_text`, show `text`)
    title_7_sections.jsonl  one record per section, full text (for parent expansion)
    clean_report.json       statistics + warnings

What it fixes
-------------
1. Fragments created by the original splitter at in-text cross-references,
   e.g. "...subdivision (d) of section eleven hundred eleven..." was cut at "(d)"
   and the "(d)" marker was dropped. We rebuild each section's full text in file
   order and put every dropped "(x)" marker back.
2. Non-unique ids  -> ids are now VAT_<section_key>_<subsection|intro>_<part>.
3. Section numbers -> "110"/"111" become "1110"/"1111"; the real statute number
   (e.g. "1111-b") is parsed from the "§ ..." line and kept for citations.
4. Wrong section_title in Articles 30+ (it contained the "Vehicle & Traffic (VAT)
   CHAPTER 71..." boilerplate line) -> parsed again from the header.
5. Tiny fragments are merged; long subsections are split at paragraph borders.
6. Adds: citation, doc_type, article_topic, cross-references, embed_text header.

Note: "section_key" keeps the dataset's variant suffix (e.g. 1111-B2). NY law
publishes several versions of some sections (effective dates), so the same
statute number can appear more than once; the key keeps them apart.
"""
import argparse
import collections
import json
import re
from pathlib import Path

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# Approximate topics per article (descriptive tags, adjust to taste).
ARTICLE_TOPIC = {
    "24": "traffic_control_devices_and_signals",
    "25": "driving_on_right_overtaking_lanes",
    "26": "right_of_way",
    "27": "pedestrians",
    "28": "turning_and_signals",
    "29": "special_stops_rail_school_bus_stop_signs",
    "30": "speed",
    "31": "alcohol_and_drugs",
    "32": "stopping_standing_parking",
    "33": "miscellaneous_rules_of_the_road",
}

# Sections whose title matches are camera / owner-liability programs and
# administrative procedure, not the basic driving rules. They are kept but
# tagged so you can filter or down-weight them at retrieval time.
ENFORCEMENT_TITLE_RE = re.compile(r"^owner liability", re.I)
PROCEDURE_ARTICLES = {"31"}          # sanctions, testing, rehab programs...
PROCEDURE_SECTIONS = {"1203-A", "1203-B", "1203-C", "1203-D", "1203-E",
                      "1203-F", "1203-G", "1203-H", "1204"}

BOILER_RE = re.compile(r"^Vehicle & Traffic \(VAT\) CHAPTER", re.I)
SECTION_LINE_RE = re.compile(r"^SECTION\s+\S+$")
STATUTE_NO_RE = re.compile(r"§\s*(\d{3,4}(?:-[A-Za-z]\d*)?)\.")
MARKER_RE = re.compile(r"\(([a-z])\)(?=[\s,.;:)]|$)")

UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def collapse_wraps(s: str) -> str:
    """Join hard-wrapped lines but keep paragraph breaks."""
    s = s.replace("\r", "")
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def parse_header(text: str):
    """Return (title, statute_number, body) from a section's first chunk."""
    lines = text.split("\n")
    title = None
    for i, ln in enumerate(lines[:6]):
        if BOILER_RE.match(ln.strip()):
            if i > 0:
                title = lines[i - 1].strip()
            break
    m = STATUTE_NO_RE.search(text)
    statute_no = m.group(1) if m else None
    # body = everything from the "§" line on (drops SECTION / title / VAT lines)
    pos = text.find("§")
    star = text.rfind("*", 0, pos)
    if star != -1 and text[star + 1:pos].strip() == "":
        pos = star
    body = text[pos:] if pos != -1 else text
    return title, statute_no, body


def parse_number_words(tokens, i):
    """Parse 'eleven hundred twelve' / 'eleven hundred eleven-a' at tokens[i:].
    Returns (section_str, next_index) or (None, i)."""
    n = len(tokens)
    j = i
    if j >= n:
        return None, i
    first = tokens[j].split("-")[0]
    if first not in UNITS or j + 1 >= n or tokens[j + 1] != "hundred":
        return None, i
    val = UNITS[first] * 100
    j += 2
    if j < n and tokens[j] == "and":
        j += 1
    suffix = ""
    if j < n:
        parts = tokens[j].split("-")
        w = parts[0]
        if w in UNITS:
            val += UNITS[w]
            if len(parts) > 1 and re.fullmatch(r"[a-z]\d*", parts[1]):
                suffix = parts[1]
            j += 1
        elif w in TENS:
            val += TENS[w]
            j += 1
            if len(parts) > 1:                       # e.g. forty-four / eighty-a
                if parts[1] in UNITS:
                    val += UNITS[parts[1]]
                elif re.fullmatch(r"[a-z]\d*", parts[1]):
                    suffix = parts[1]
            elif j < n and tokens[j] in UNITS and UNITS[tokens[j]] < 10:
                val += UNITS[tokens[j]]
                j += 1
    sec = str(val)
    if suffix:
        sec += "-" + suffix.upper()
    return sec, j


def extract_references(text: str, known_bases: set):
    flat = re.sub(r"\s+", " ", text.lower())
    refs = set()
    for m in re.finditer(r"\bsections?\s+(\d{3,4}(?:-[a-z]\d?)?)\b", flat):
        refs.add(m.group(1).upper())
    for m in re.finditer(r"\bsections?\s+([a-z\- ,]+)", flat):
        toks = [t.strip(",;.") for t in m.group(1).split()]
        i = 0
        while i < len(toks):
            sec, j = parse_number_words(toks, i)
            if sec:
                refs.add(sec)
                i = j
                if i < len(toks) and toks[i] in ("and", "or"):
                    i += 1
                    continue
                break
            else:
                break
    return sorted(r for r in refs if r in known_bases)


def split_paragraphs(text: str, max_chars: int):
    """Split long text at paragraph (then sentence) boundaries."""
    if len(text) <= max_chars:
        return [text]
    paras = text.split("\n\n")
    pieces = []
    for p in paras:
        if len(p) <= max_chars:
            pieces.append(p)
        else:
            sents = re.split(r"(?<=[.;:])\s+", p)
            cur = ""
            for s in sents:
                if cur and len(cur) + len(s) + 1 > max_chars:
                    pieces.append(cur)
                    cur = s
                else:
                    cur = (cur + " " + s).strip()
            if cur:
                pieces.append(cur)
    out, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------
# Step 1: rebuild full text of every section
# --------------------------------------------------------------------------
def rebuild_sections(rows):
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault(r["metadata"]["section"], []).append(r)

    sections = []
    for old_key, chunks in groups.items():
        first = chunks[0]
        title, statute_no, body = parse_header(first["text"])
        title = title or first["metadata"]["section_title"]
        full = collapse_wraps(body)
        n_marker_fixed = 0
        for r in chunks[1:]:
            letter = r["metadata"]["subsection"]
            frag = collapse_wraps(r["text"])
            if not frag:
                continue
            if letter and not frag.startswith(f"({letter})"):
                glue = "" if frag[0] in ",.;:)" else " "
                frag = f"({letter}){glue}{frag}"
                n_marker_fixed += 1
            # new paragraph if the previous text ended a sentence/clause,
            # otherwise it was a cut in the middle of a sentence
            sep = "\n\n" if re.search(r"[.:;]$", full) else " "
            full = f"{full}{sep}{frag}"
        key = old_key
        if re.fullmatch(r"\d{3}", old_key):           # 110 -> 1110, 111 -> 1111
            key = "1" + old_key
        base_num = key.split("-")[0]
        if statute_no and statute_no.split("-")[0] != base_num:
            print(f"  ! section number mismatch: file={old_key} header={statute_no}")
        sections.append({
            "section_key": key,
            "old_section": old_key,
            "statute_no": (statute_no or key).upper(),
            "title": title,
            "article": first["metadata"]["article"],
            "text": full,
            "orig_ids": [c["id"] for c in chunks],
            "orig_chars": sum(len(c["text"]) for c in chunks),
            "markers_restored": n_marker_fixed,
        })
    return sections


# --------------------------------------------------------------------------
# Step 2: split a section at its top-level (a), (b), (c) ... subdivisions
# --------------------------------------------------------------------------
def split_subdivisions(full: str):
    """Return [(letter|None, text), ...]. Letters must run in order (a->b->c,
    one gap allowed because the statute itself skips some, e.g. 1111 has no (c)),
    which keeps nested lists like '(i) ...' from being mistaken for new subdivisions."""
    cuts = []                       # (position, letter)
    cur = None
    for m in MARKER_RE.finditer(full):
        L = m.group(1)
        pos = m.start()
        at_para_start = full[max(0, pos - 2):pos] == "\n\n"
        inline_a = (L == "a" and cur is None and pos < 400)
        if not (at_para_start or inline_a):
            continue
        if cur is None:
            if L == "a":
                cuts.append((pos, L)); cur = L
        elif 1 <= ord(L) - ord(cur) <= 2:
            cuts.append((pos, L)); cur = L
    if not cuts:
        return [(None, full)]
    parts = []
    if cuts[0][0] > 0:
        pre = full[:cuts[0][0]].strip()
        if pre:
            parts.append((None, pre))
    for idx, (pos, L) in enumerate(cuts):
        end = cuts[idx + 1][0] if idx + 1 < len(cuts) else len(full)
        parts.append((L, full[pos:end].strip()))
    return parts


# --------------------------------------------------------------------------
# Step 3: build final chunks
# --------------------------------------------------------------------------
def build_chunks(sections, max_chars, min_chars):
    known_bases = {s["statute_no"].upper() for s in sections}
    known_bases |= {s["section_key"].upper() for s in sections}
    chunks = []
    for s in sections:
        parts = split_subdivisions(s["text"])
        # merge tiny leading intro into the next part
        if len(parts) > 1 and parts[0][0] is None and len(parts[0][1]) < min_chars:
            nxt = parts[1]
            parts = [(nxt[0], parts[0][1] + "\n\n" + nxt[1])] + parts[2:]
        # merge tiny subdivisions into the previous one
        merged = []
        for L, t in parts:
            if merged and len(t) < min_chars and L is not None:
                pl, pt = merged[-1]
                merged[-1] = (pl, pt + "\n\n" + t)
            else:
                merged.append((L, t))
        parts = merged

        title = s["title"]
        art = s["article"]
        if ENFORCEMENT_TITLE_RE.match(title or ""):
            doc_type = "enforcement_program"
        elif art in PROCEDURE_ARTICLES or s["section_key"].upper() in PROCEDURE_SECTIONS:
            doc_type = "penalty_or_procedure"
        else:
            doc_type = "substantive_rule"

        for L, t in parts:
            pieces = split_paragraphs(t, max_chars)
            # avoid a tiny last piece
            if len(pieces) > 1 and len(pieces[-1]) < min_chars:
                pieces[-2] = pieces[-2] + "\n\n" + pieces[-1]
                pieces.pop()
            for k, piece in enumerate(pieces):
                sub_label = f"({L})" if L else ""
                shown = re.sub(r"-([A-Z]\d*)", lambda m: "-" + m.group(1).lower(),
                               s["statute_no"])           # 1111-B -> 1111-b
                citation = f"VTL § {shown}{sub_label}"
                header = (f"NY Vehicle & Traffic Law {citation} - {title}"
                          + (f" [part {k + 1}/{len(pieces)}]" if len(pieces) > 1 else ""))
                refs = [r for r in extract_references(piece, known_bases)
                        if r != s["statute_no"].upper() and r != s["section_key"].upper()]
                chunks.append({
                    "id": f"VAT_{s['section_key']}_{L or 'intro'}_{k}",
                    "text": piece,
                    "embed_text": f"{header}\n{piece}",
                    "metadata": {
                        "source": "NYS Open Legislation",
                        "law_code": "VAT",
                        "law_name": "Vehicle and Traffic Law",
                        "jurisdiction": "NYS",
                        "title": "7",
                        "article": art,
                        "article_topic": ARTICLE_TOPIC.get(art, "other"),
                        "section_key": s["section_key"],
                        "section_number": s["statute_no"],
                        "section_title": title,
                        "subsection": L,
                        "part": k,
                        "n_parts": len(pieces),
                        "citation": citation,
                        "doc_type": doc_type,
                        "references": refs,
                        "char_len": len(piece),
                    },
                })
    return chunks


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out_dir")
    ap.add_argument("--max-chars", type=int, default=3500)
    ap.add_argument("--min-chars", type=int, default=150)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    print(f"read {len(rows)} chunks")

    sections = rebuild_sections(rows)
    print(f"rebuilt {len(sections)} sections")
    chunks = build_chunks(sections, a.max_chars, a.min_chars)

    # ---- validation ------------------------------------------------------
    ids = collections.Counter(c["id"] for c in chunks)
    dups = [i for i, n in ids.items() if n > 1]
    empty = [c["id"] for c in chunks if not c["text"].strip()]
    n_per_sec = collections.Counter(c["metadata"]["section_key"] for c in chunks)
    short = [c["id"] for c in chunks if c["metadata"]["char_len"] < a.min_chars
             and n_per_sec[c["metadata"]["section_key"]] > 1]   # whole tiny sections are fine
    lens = [c["metadata"]["char_len"] for c in chunks]
    old_chars = sum(s["orig_chars"] for s in sections)
    new_chars = sum(len(s["text"]) for s in sections)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "title_7_clean.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out / "title_7_sections.jsonl", "w", encoding="utf-8") as f:
        for s in sections:
            rec = {"section_key": s["section_key"], "section_number": s["statute_no"],
                   "title": s["title"], "article": s["article"], "text": s["text"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "input_chunks": len(rows),
        "output_chunks": len(chunks),
        "sections": len(sections),
        "unique_ids_before": len({r['id'] for r in rows}),
        "unique_ids_after": len(ids),
        "duplicate_ids_after": dups,
        "empty_chunks": empty,
        "chunks_below_min_chars": short,
        "char_len_min_avg_max": [min(lens), sum(lens) // len(lens), max(lens)],
        "markers_restored": sum(s["markers_restored"] for s in sections),
        "chars_before_vs_after_rebuild": [old_chars, new_chars],
        "doc_type_counts": dict(collections.Counter(c["metadata"]["doc_type"] for c in chunks)),
        "chunks_with_references": sum(1 for c in chunks if c["metadata"]["references"]),
    }
    (out / "clean_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert not dups, "duplicate ids!"
    assert not empty, "empty chunks!"


if __name__ == "__main__":
    main()
