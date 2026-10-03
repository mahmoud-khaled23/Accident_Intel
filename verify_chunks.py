"""
verify_chunks.py
----------------
Checks the output of build_chunks.py and prints a report of possible problems.

    python verify_chunks.py --data ./data --out ./out     # full report
    python verify_chunks.py --out ./out --show 1180       # show one section and its chunks
    python verify_chunks.py --out ./out --random 3        # show 3 random sections for eyeballing
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from build_chunks import extract_sections, info_from_path, resolve_ref, clean_lines, SECTION_START_RE

# Website text that should never appear inside a chunk
LEAK_RE = re.compile(
    r"find your senator|new york state senate|twitter icon|facebook|"
    r"viewing most recent revision|view historical revision|search openlegislation|"
    r"^\s*SECTION\s+\S+|^\s*(previous|next)\s*$", re.I | re.M)

problems = Counter()


def flag(kind, msg):
    problems[kind] += 1
    if problems[kind] <= 10:           # only print the first 10 examples of each kind
        print(f"  [{kind}] {msg}")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def body_of(text):
    """Text without its first line (the header)."""
    return text.split("\n", 1)[1] if "\n" in text else ""


def load(out):
    out = Path(out)
    parents = json.loads((out / "parents.json").read_text(encoding="utf-8"))
    children = [json.loads(l) for l in open(out / "children.jsonl", encoding="utf-8")]
    return parents, children


def show(parents, children, num):
    pid = f"VTL-{num.lower()}"
    if pid not in parents:
        print(f"{pid} not found. Some existing ids: {list(parents)[:8]}")
        return
    p = parents[pid]
    print("=" * 80)
    print(f"PARENT {pid} | article={p['article']} | category={p['category']} | refs={p['refs']}")
    print(f"source file: {p.get('source', '?')}")
    print(f"words={len(p['text'].split())}")
    print("-" * 80)
    print(p["text"][:1500] + (" ..." if len(p["text"]) > 1500 else ""))
    kids = [c for c in children if c["parent_id"] == pid]
    print("=" * 80)
    print(f"{len(kids)} children:")
    for c in kids:
        w = len(c["text"].split())
        print(f"\n--- {c['id']}  ({w} words) ---")
        print(c["text"][:350] + (" ..." if len(c["text"]) > 350 else ""))


MARK_LINE = re.compile(r"^\s*(\([a-z]{1,4}\)|\d+\.)(\s|$)")


def markers(parents, out, num):
    """Debug: list every line that starts with (a)/(b)/1./2. in the RAW file and in the cleaned parent."""
    pid = f"VTL-{num.lower()}"
    if pid not in parents:
        print(f"{pid} not found"); return
    p = parents[pid]
    meta = Path(out) / "meta.json"
    if meta.exists() and p.get("source"):
        raw = Path(json.loads(meta.read_text(encoding="utf-8"))["data"]) / p["source"]
        print(f"RAW FILE: {raw}")
        for i, ln in enumerate(raw.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if MARK_LINE.match(ln):
                print(f"  {i:4d}: {ln.strip()[:90]}")
    print(f"\nPARENT {pid} AFTER CLEANING (lines starting with a marker):")
    for i, ln in enumerate(body_of(p["text"]).split("\n"), 1):
        if MARK_LINE.match(ln):
            print(f"  {i:4d}: {ln.strip()[:90]}")


def check_coverage(data, parents):
    print("\n[1] Coverage: did every file / '§' become a parent?")
    files = sorted(Path(data).rglob("*.txt"))
    if not files:
        print(f"  WARNING: no .txt files found under {Path(data).resolve()}")
        print("  Use the SAME --data path you passed to build_chunks.py.")
        return
    expected = set()
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        _, hint = info_from_path(f, Path(data))
        found = extract_sections(raw, hint)
        if not found:
            flag("FILE_SKIPPED", f"{f}  (no '§' and no section number in its path)")
        n_marks = sum(1 for ln in clean_lines(raw) if SECTION_START_RE.search(ln))
        if found and found[0][2] and len(found) != n_marks:
            flag("SECTION_COUNT", f"{f}: {n_marks} '§' markers but {len(found)} sections extracted")
        for num, _, _ in found:
            expected.add(f"VTL-{num}")
    for m in sorted(expected - set(parents)):
        flag("MISSING_PARENT", m)
    print(f"  files: {len(files)} | expected sections: {len(expected)} | actual parents: {len(parents)}")


def check_lossless(parents, children):
    print("\n[2] No text lost: do the children add up to the parent text?")
    by_parent = {}
    for c in children:
        by_parent.setdefault(c["parent_id"], []).append(c)
    for pid, p in parents.items():
        kids = by_parent.get(pid, [])
        if not kids:
            flag("NO_CHILDREN", pid); continue
        words = Counter()
        ctxs = set()
        for c in kids:
            body = body_of(c["text"])
            ctx = c.get("ctx", "")
            if ctx:
                ctxs.add(ctx)
                body = body.replace(ctx, "", 1)       # lead-in is repeated on purpose
            words.update(norm(body).split())
        for ctx in ctxs:                              # count each lead-in once
            words.update(norm(ctx).split())
        if words != Counter(norm(body_of(p["text"])).split()):
            diff = sum((Counter(norm(body_of(p["text"])).split()) - words).values())
            extra = sum((words - Counter(norm(body_of(p["text"])).split())).values())
            flag("TEXT_MISMATCH", f"{pid}: {diff} words missing from children, {extra} extra")


def check_chunks(parents, children):
    print("\n[3] Chunk quality")
    ids = Counter(c["id"] for c in children)
    for i, n in ids.items():
        if n > 1:
            flag("DUP_ID", i)
    words = []
    for c in children:
        body = body_of(c["text"])
        w = len(c["text"].split())
        words.append(w)
        if w > 600:
            flag("TOO_BIG", f"{c['id']}: {w} words")
        if len(body.split()) < 8:
            flag("TOO_SMALL", f"{c['id']}: '{norm(body)[:60]}'")
        if LEAK_RE.search(c["text"]):
            flag("BOILERPLATE_LEAK", f"{c['id']}: {LEAK_RE.search(c['text']).group(0)!r}")
        if c.get("ctx") and body.startswith(c["ctx"]):
            body = body[len(c["ctx"]):].lstrip()      # skip the repeated lead-in
        m = re.fullmatch(r"([a-z])", c["sub"])    # a chunk labelled "b" must start with "(b)"
        if m and not body.lstrip().startswith(f"({m.group(1)})"):
            flag("BAD_START", f"{c['id']} does not start with ({m.group(1)})")
    words.sort()
    print(f"  chunks={len(children)} | words min={words[0]} median={words[len(words)//2]} max={words[-1]}")


def check_structure(parents, children):
    print("\n[4] Structure: unsplit long sections / unused subdivision markers")
    kids = {}
    for c in children:
        kids.setdefault(c["parent_id"], []).append(c)
    for pid, p in parents.items():
        body = body_of(p["text"])
        nwords = len(body.split())
        ks = kids.get(pid, [])
        if nwords > 450 and len(ks) == 1:
            flag("LONG_NOT_SPLIT", f"{pid}: {nwords} words in a single chunk (no (a)(b) found?)")
        letters_in_body = re.findall(r"(?m)^\(([a-z])\)\s", body)
        core = [l for l in letters_in_body if l not in "ivx"]   # i / v / x are usually roman numerals
        if core:                              # a letter that never appears between the first and last one
            seen = {ord(l) - 97 for l in core}
            gaps = [chr(97 + k) for k in range(min(seen), max(seen))
                    if k not in seen and chr(97 + k) not in "ivx"]
            if gaps:
                flag("LETTER_GAP",
                     f"{pid}: the source goes ({', '.join(chr(97 + k) for k in sorted(seen))}) "
                     f"and never has ({', '.join(gaps)}) - check the source file is complete")
        used = {c["sub"].split(".")[0].split("~")[0] for c in ks}
        used_letters = [u for u in used if len(u) == 1 and u.isalpha()]
        if len(ks) > 1 and used_letters:
            top = max(used_letters)
            # Letters AFTER the last chunk letter (ignoring roman-numeral-like i/v/x) suggest
            # that real top-level subdivisions were missed. Letters before it are normally
            # just nested lists that restart at (a) - those are fine.
            missed = sorted({l for l in letters_in_body if l > top and l not in "ivx"})
            if missed:
                flag("MISSED_SUBDIVISIONS",
                     f"{pid}: chunks end at ({top}) but ({', '.join(missed)}) also appear at line starts "
                     f"- check with --show {pid[4:]}")
        if not p["title"]:
            flag("NO_TITLE", pid)


def check_refs(parents):
    print("\n[5] Cross references pointing to sections that are not in your data")
    missing = Counter()
    for p in parents.values():
        for r in p["refs"]:
            if not resolve_ref(r, parents):
                missing[r] += 1
    if missing:
        print(f"  {len(missing)} referenced sections are missing (not necessarily wrong - "
              f"they may be outside articles 24-33). Top 10:")
        print("  ", missing.most_common(10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--show")
    ap.add_argument("--markers", help="debug: list marker lines of one section, raw vs cleaned")
    ap.add_argument("--random", type=int, default=0)
    a = ap.parse_args()

    parents, children = load(a.out)
    if a.show:
        return show(parents, children, a.show)
    if a.markers:
        return markers(parents, a.out, a.markers)
    if a.random:
        for pid in random.sample(list(parents), min(a.random, len(parents))):
            show(parents, children, pid.replace("VTL-", ""))
        return

    print(f"parents={len(parents)} children={len(children)}")
    data = a.data
    meta = Path(a.out) / "meta.json"
    if meta.exists() and (not data or not list(Path(data).rglob("*.txt"))):
        data = json.loads(meta.read_text(encoding="utf-8"))["data"]
        print(f"(using the data folder recorded by build_chunks.py: {data})")
    if data:
        check_coverage(data, parents)
    check_lossless(parents, children)
    check_chunks(parents, children)
    check_structure(parents, children)
    check_refs(parents)

    print("\n" + "=" * 60)
    if not problems:
        print("OK - no problems found")
    else:
        print("Problems summary:", dict(problems))


if __name__ == "__main__":
    main()