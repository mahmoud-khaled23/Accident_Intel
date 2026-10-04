"""
build_chunks.py
---------------
Reads the traffic-law text files (e.g. data/<article>/<section>/text.txt) and produces:
  - out/parents.json    : one entry per full section (this is what gets sent to the LLM)
  - out/children.jsonl  : small chunks (these are the ones that get embedded)

Usage:
    python build_chunks.py --data ./data --out ./out
"""
import argparse
import json
import re
import string
from pathlib import Path

# ----------------------------- Settings -----------------------------
MAX_WORDS_PARENT = 400   # a section at or below this size stays as a single chunk
MAX_WORDS_SUB = 500      # a subdivision longer than this is split on "1. 2. 3."
MAX_WORDS_HARD = 450     # max chunk size used by the last-resort fallback splitter
INTRO_CTX_MAX_WORDS = 40 # a short lead-in (e.g. '(b) Yellow indications:') is repeated in every chunk below it

# Sections that don't help decide fault (owner-liability fines, camera programs).
# They are kept in the data but tagged, so retrieval can filter them out.
EXCLUDE_PATTERNS = re.compile(
    r"owner liability|demonstration program|photo violation-monitoring|"
    r"speed violation monitoring|bus lane restrictions",
    re.I,
)

# ----------------------------- Cleaning -----------------------------
# Lines copied from the website that are not part of the law text.
BOILERPLATE_LINES = [
    r"the new york state senate", r"find your senator", r"legislation",
    r"search openlegislation statutes", r"previous", r"next", r"up",
    r"share", r"facebook", r"twitter icon", r"email",
    r"view historical revision as of:.*", r"viewing most recent revision.*",
    r"\d{4}-\d{2}-\d{2}",
    r"the laws of new york.*",
    r"vehicle & traffic \(vat\) chapter.*",
    r"title vii", r"rules of the road",
]
BOILER_RE = re.compile(r"^\s*(?:%s)\s*$" % "|".join(BOILERPLATE_LINES), re.I)

SECTION_HEADER_RE = re.compile(r"^\s*SECTION\s+\S+", re.I)                        # "SECTION 1121"
# "§ 1121. ..." or "§ 1111-c-1. ..."  - may also appear MID-LINE after a "*" (copy/paste artefact:
# "Title text * § 1111-c-1. Title text. (a) ...")
SECTION_START_RE = re.compile(r"(?:^\s*|\*\s*)§\s*(\d+(?:-[a-zA-Z])?(?:-\d{1,2})?)\*?\.\s*(.*)$")
MARKER_RE = re.compile(r"^(\([a-z]{1,4}\)|\d+\.)\s")                              # "(a) " or "1. "


def clean_lines(text):
    return [ln for ln in text.splitlines() if not BOILER_RE.match(ln)]


SENTENCE_END_RE = re.compile(r"[.:;][\"')\]]*(?:\s+(?:and|or))?\s*$", re.I)


def reflow(text):
    """
    Join hard-wrapped lines into paragraphs.
    A marker like "(d)" or "2." starts a NEW paragraph unless it is a wrapped in-sentence
    reference, e.g.  "... pursuant to subdivision\n(d) of this section".
    A line is treated as such a reference only when the previous line did NOT end a
    sentence AND the text after the marker starts in lowercase ("of this section").
    Real subdivisions start with a capital ("(a) Traffic, except pedestrians...").
    """
    paras, cur = [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        m = MARKER_RE.match(s)
        if m and cur:
            is_reference = s[m.end():m.end() + 1].islower() and not SENTENCE_END_RE.search(cur[-1])
            if not is_reference:
                paras.append(" ".join(cur)); cur = []
        cur.append(s)
    if cur:
        paras.append(" ".join(cur))
    return "\n".join(paras)


# ----------------------------- Extract sections from a file -----------------------------
def extract_sections(raw_text, hint_num=None):
    """
    Returns a list of (section_number, full_text, has_title).
    If the file has no "§ <number>." marker, the whole file is treated as ONE section
    and its number is taken from hint_num (derived from the folder/file name).
    """
    sections, cur_num, cur_lines = [], None, []

    def close():
        nonlocal cur_num, cur_lines
        if cur_num and cur_lines:
            sections.append((cur_num, "\n".join(cur_lines), True))
        cur_num, cur_lines = None, []

    lines = clean_lines(raw_text)
    for ln in lines:
        if SECTION_HEADER_RE.match(ln):      # website block header: ends the previous section
            close()
            continue
        m = SECTION_START_RE.search(ln)
        if m:                                # a real section starts here
            close()
            cur_num = m.group(1).lower()
            cur_lines = [m.group(2)]
            continue
        if cur_num:
            cur_lines.append(ln)
    close()

    if not sections and hint_num:            # no "§" found -> whole file is one section
        body = "\n".join(ln for ln in lines if not SECTION_HEADER_RE.match(ln)).strip()
        if body:
            sections.append((hint_num, body, False))
    return sections


def split_title(text):
    """Split the title from the body: 'Title. (a) body...'"""
    text = reflow(text).strip()
    m = re.match(r"(.+?)\.(?:\s+|$)", text, re.S)
    if m and len(m.group(1)) <= 250:
        return m.group(1).strip(), text[m.end():].strip()
    return "", text


# ----------------------------- Cross references -----------------------------
# The statute writes references in words: "section eleven hundred eleven" -> 1111
_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: (i + 2) * 10 for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
_NW = "|".join(sorted(list(_ONES) + list(_TENS), key=len, reverse=True))
_NUMPH = rf"(?:{_NW})(?:[-\s](?:{_NW}))?\b"
WORD_REF = re.compile(
    rf"\bsections?\s+({_NUMPH})\s+hundred(?:\s+(?:and\s+)?({_NUMPH}))?(?:-([a-z])\b)?", re.I)
DIGIT_REF = re.compile(r"\bsections?\s+(\d{3,4}(?:-[a-z])?)\b", re.I)


def _words_to_num(s):
    return sum(_TENS.get(t, _ONES.get(t, 0)) for t in re.split(r"[-\s]+", s.lower()) if t)


def extract_refs(text, self_num):
    refs = set()
    for m in WORD_REF.finditer(text):
        n = _words_to_num(m.group(1)) * 100 + (_words_to_num(m.group(2)) if m.group(2) else 0)
        refs.add(f"{n}-{m.group(3).lower()}" if m.group(3) else str(n))
    for m in DIGIT_REF.finditer(text):
        refs.add(m.group(1).lower())
    refs.discard(self_num)
    return sorted(refs)


def resolve_ref(ref, parent_ids):
    """
    Map a reference like '1111-c' to existing parent ids. Some sections exist in several
    numbered versions (VTL-1111-c-1, VTL-1111-c-2), so a reference may match several.
    """
    rid = f"VTL-{ref}"
    if rid in parent_ids:
        return [rid]
    return sorted(i for i in parent_ids if re.fullmatch(re.escape(rid) + r"-\d+", i))


# ----------------------------- Subdivision splitting -----------------------------
LETTER_RE = re.compile(r"^\(([a-z])\)\s")
NUMBER_RE = re.compile(r"^(\d+)\.\s")


def _letter_starts(lines):
    """
    Find the line indexes where TOP-LEVEL subdivisions (a), (b), (c)... begin.
    Statutes often nest lettered lists inside numbered paragraphs that restart at (a):
        (a) Whenever ... as follows:      <- top level
        1. Green indication
        (a) Traffic ...  (b) ...          <- nested, restarts at (a)
        2. Steady yellow
        (a) ...                           <- nested again
        (b) No pedestrian shall ...       <- back to top level
    We keep a stack of letter lists: a marker continues the deepest list it can
    (next letter), an "(a)" opens a nested list, and a numbered paragraph ("2.") closes
    all nested lists. Only markers that land on the top-level list start a chunk.
    """
    stack, starts = [], []          # stack[k] = index (0=a, 1=b...) of last letter at depth k
    for i, ln in enumerate(lines):
        s = ln.strip()
        if NUMBER_RE.match(s):
            del stack[1:]           # a new numbered paragraph ends any nested letter list
            continue
        m = LETTER_RE.match(s)
        if not m:
            continue
        idx = string.ascii_lowercase.index(m.group(1))
        rest = s[m.end():]
        target = None
        for depth in range(len(stack) - 1, -1, -1):        # 1) exact next letter, deepest list first
            if idx == stack[depth] + 1:
                target = depth
                break
        if target is None and rest[:1].isupper():          # 2) tolerate a small gap: some sources
            for depth in range(len(stack) - 1, -1, -1):    #    skip a letter, e.g. (a) (b) (d)
                if stack[depth] < idx <= stack[depth] + 3:
                    target = depth
                    break
        if target is not None:
            del stack[target + 1:]
            stack[target] = idx
            if target == 0:
                starts.append((i, m.group(1)))
        elif idx == 0:                                     # "(a)" again -> a nested list begins
            stack.append(0)
            if len(stack) == 1:
                starts.append((i, "a"))
        # any other letter is a stray reference -> ignored
    return starts


def _number_starts(lines):
    starts, nxt = [], 1
    for i, ln in enumerate(lines):
        m = NUMBER_RE.match(ln.strip())
        if m and int(m.group(1)) == nxt:
            starts.append((i, m.group(1)))
            nxt += 1
    return starts


def split_sequential(text, kind):
    """
    kind='letter' -> split on top-level (a)(b)(c)...   |   kind='number' -> split on 1. 2. 3. ...
    Markers must come IN ORDER, which prevents cutting on in-sentence references such as
    "subdivision (d) of this section".
    """
    lines = text.split("\n")
    starts = _letter_starts(lines) if kind == "letter" else _number_starts(lines)
    if not starts:
        return [("", text.strip())]

    parts = []
    intro = "\n".join(lines[: starts[0][0]]).strip()
    if intro:
        parts.append(("intro", intro))
    for k, (idx, label) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        parts.append((label, "\n".join(lines[idx:end]).strip()))
    return parts


def take_intro(parts):
    """If the first part is a SHORT lead-in line, return (remaining_parts, lead_in_text)."""
    if len(parts) > 1 and parts[0][0] == "intro" and len(parts[0][1].split()) <= INTRO_CTX_MAX_WORDS:
        return parts[1:], parts[0][1]
    return parts, ""


def hard_split(text, max_words=MAX_WORDS_HARD):
    """Last-resort fallback: pack sentences/paragraphs up to max_words per chunk."""
    units = [u for p in text.split("\n") for u in re.split(r"(?<=[.;])\s+", p) if u.strip()]
    out, cur, n = [], [], 0
    for u in units:
        w = len(u.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(u); n += w
    if cur:
        out.append(" ".join(cur))
    return out


# ----------------------------- Build parent and children -----------------------------
def build(article, num, full_text, has_title=True, source="", version=1):
    if has_title:
        title, body = split_title(full_text)
    else:                                    # file had no "§": accept a title only if a marker follows it
        title, body = split_title(full_text)
        if not (title and MARKER_RE.match(body)):
            title, body = "", reflow(full_text).strip()
    sid = f"VTL-{num}"
    header = f"NY VTL § {num}" + (f" - {title}" if title else "")
    category = "enforcement_program" if EXCLUDE_PATTERNS.search(f"{title} {body}") else "core"

    parent = {
        "id": sid, "article": article, "section": num, "title": title,
        "category": category, "refs": extract_refs(body, num), "source": source, "version": version,
        "text": f"{header}\n{body}",
    }

    pieces = []  # (label, text, ctx)  ctx = short lead-in repeated at the top of each chunk
    if len(body.split()) <= MAX_WORDS_PARENT:
        pieces.append(("full", body, ""))
    else:
        top, top_ctx = take_intro(split_sequential(body, "letter"))
        for lab, txt in top:
            if len(txt.split()) > MAX_WORDS_SUB:
                subs, sub_ctx = take_intro(split_sequential(txt, "number"))
                if subs and subs[0][0] != "":          # a real 1. 2. 3. split was found
                    ctx = "\n".join(x for x in (top_ctx, sub_ctx) if x)
                    for lab2, t2 in subs:
                        pieces.append((f"{lab}.{lab2}" if lab else lab2, t2, ctx))
                    continue
            pieces.append((lab or "body", txt, top_ctx))

    # Any piece that is still too large goes through the fallback splitter.
    children, seen = [], {}
    for lab, txt, ctx in pieces:
        chunks = [txt] if len(txt.split()) <= MAX_WORDS_HARD * 1.2 else hard_split(txt)
        for j, c in enumerate(chunks):
            label = lab if len(chunks) == 1 else f"{lab}~{j + 1}"
            seen[label] = seen.get(label, 0) + 1
            if seen[label] > 1:
                label = f"{label}_{seen[label]}"
            shown = "" if label == "full" else f" ({label})"
            children.append({
                "id": f"{sid}#{label}", "parent_id": sid, "article": article,
                "section": num, "sub": label, "category": category, "ctx": ctx,
                "text": f"{header}{shown}\n" + (f"{ctx}\n" if ctx else "") + c,
            })
    return parent, children


# ----------------------------- Article / section number from the path -----------------------------
# 1121 / 1111-b / 1111-c-1  (a trailing version digit like in "1111-B2" or "1111-B*2" is allowed but not captured)
SEC_NUM_RE = re.compile(r"(?<!\d)(\d{3,4}(?:-[A-Za-z])?(?:-\d{1,2})?)(?:\*?\d{1,2})?(?![\d])")
# version of a section that exists in several amended versions: "1111-B2" / "1111-B*3" / "1111*2" -> 2 / 3 / 2
VERSION_RE = re.compile(r"\d{3,4}(?:-[A-Za-z](?:\*?(\d{1,2})(?!\d))|\*(\d{1,2})(?!\d))")
ART_RE = re.compile(r"article\D{0,3}(\d{1,3}(?:-[A-Za-z])?)", re.I)


def section_from_name(name):
    """'Section 1121' / '1111-B*3' / '1111-b' -> '1111-b'"""
    m = SEC_NUM_RE.search(name)
    return m.group(1).lower() if m else None


def version_from_path(path, data_root):
    """Version number from the folder/file name ('1111-B3' -> 3). No marker means version 1."""
    rel = path.relative_to(data_root)
    for name in list(reversed(rel.parts[:-1])) + [path.stem]:
        m = VERSION_RE.search(name)
        if m:
            return int(m.group(1) or m.group(2))
    return 1


def info_from_path(path, data_root):
    """Extract (article, section_hint) from the folder names and the file name."""
    rel = path.relative_to(data_root)
    dirs, stem = list(rel.parts[:-1]), path.stem
    article = None
    for d in dirs:                           # folder like "Article 24"
        m = ART_RE.search(d)
        if m:
            article = m.group(1).upper(); break
    if article is None and dirs:             # folder named with a plain number, e.g. "24"
        m = re.fullmatch(r"\D*(\d{1,2}(?:-[A-Za-z])?)\D*", dirs[0])
        if m:
            article = m.group(1).upper()
    hint = None
    for name in reversed(dirs):              # closest folder to the file first
        hint = section_from_name(name)
        if hint:
            break
    if not hint:
        hint = section_from_name(stem)       # then the file name
    return article, hint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./out")
    args = ap.parse_args()

    data_root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parents, children, dup = {}, {}, []
    n_fallback = 0
    files = sorted(data_root.rglob("*.txt"))
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        article, hint = info_from_path(f, data_root)
        if article is None:
            m = re.search(r"ARTICLE\s+(\d+(?:-[A-Za-z])?)", raw, re.I)
            article = m.group(1).upper() if m else "unknown"
        found = extract_sections(raw, hint)
        if not found:
            print(f"[warn] file has no '§' and no section number in its path: {f}")
        elif found[0][2] is False:
            n_fallback += 1
        for num, text, has_title in found:
            p, ch = build(article, num, text, has_title, str(f.relative_to(data_root)),
                          version_from_path(f, data_root))
            if p["id"] in parents:           # same section in several versions: keep the HIGHEST version
                old = parents[p["id"]]
                if p["version"] < old["version"]:
                    dup.append((p["id"], p["source"], old["source"], old["version"]))
                    continue
                dup.append((p["id"], old["source"], p["source"], p["version"]))
                for cid in [c["id"] for c in children.values() if c["parent_id"] == p["id"]]:
                    children.pop(cid)
            parents[p["id"]] = p
            for c in ch:
                children[c["id"]] = c

    (out / "parents.json").write_text(
        json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "children.jsonl", "w", encoding="utf-8") as fh:
        for c in children.values():
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # remember where the data lives so verify_chunks.py can find it by itself
    (out / "meta.json").write_text(
        json.dumps({"data": str(data_root.resolve())}, indent=2), encoding="utf-8")

    excluded = sum(1 for p in parents.values() if p["category"] != "core")
    print(f"files: {len(files)} | parents: {len(parents)} | children: {len(children)}")
    print(f"files without § (section number taken from folder/file name): {n_fallback}")
    print(f"excluded (enforcement_program): {excluded} | duplicates replaced: {len(dup)}")
    for pid, dropped, kept, ver in dup[:30]:
        print(f"   duplicate {pid}: kept {kept} (version {ver})  | dropped {dropped}")
    lens = sorted(len(c["text"].split()) for c in children.values())
    if lens:
        print(f"child words  min={lens[0]}  median={lens[len(lens)//2]}  max={lens[-1]}")


if __name__ == "__main__":
    main()