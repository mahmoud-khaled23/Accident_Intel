"""
analyze_accident.py
-------------------
Accident description  ->  structured facts + search queries (LLM call 1)
                      ->  relevant law sections (RAG retrieval)
                      ->  fault analysis based ONLY on those sections (LLM call 2)

Setup:
    pip install anthropic
    set ANTHROPIC_API_KEY=your_key          (Windows)   |   export ANTHROPIC_API_KEY=your_key  (Mac/Linux)

Usage:
    python analyze_accident.py --out ./out --file accident_example.txt
    python analyze_accident.py --out ./out --text "Car A ran a red light and hit car B ..." --lang ar
"""
import argparse
import json
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from index_and_retrieve import load, retrieve

DEFAULT_MODEL = "Qwen/Qwen3-8B"
MAX_WORDS_PER_LAW = 2500      # safety cap for very long sections

EXTRACT_SYSTEM = """You are a traffic-accident analyst. Read the accident description (it may be in
any language) and return ONLY a JSON object, no markdown, with these keys:

{
  "summary": "2-3 sentence neutral summary in English",
  "road_type": "intersection / highway / roundabout / parking lot / unknown",
  "traffic_control": "signal color, stop sign, yield sign, none, or unknown",
  "vehicles": [
    {"id": "A", "action": "what it did", "direction": "...", "speed": "...", "signals_used": "..."}
  ],
  "impact": "which part of which vehicle hit which part of which vehicle",
  "behaviors": ["short neutral descriptions of each notable driver behavior"],
  "unknowns": ["important facts that are NOT stated in the description"],
  "relevant_sections": ["3-6 section numbers copied from the SECTION CATALOG, e.g. \"1111\""],
  "queries": ["4-8 short English search queries written in the style of New York Vehicle and Traffic Law"]
}

Rules:
- Describe only what the text states. If something is not stated, put it in "unknowns", do not guess.
- Queries must target the LEGAL RULE for each behavior, e.g. "disobeying traffic control signal red",
  "failure to yield right of way at intersection", "unsafe turning without signal",
  "following too closely", "speed not reasonable and prudent".
- Do not mention fault or blame in the JSON.
- relevant_sections: pick, from the SECTION CATALOG in the user message, the sections whose rules govern
  the behaviors described. Prefer the GENERAL rule for each behavior (signals, obeying traffic-control
  devices, right of way, speed, lane use, turning, following distance). Do NOT pick sections about special
  situations (emergency vehicles, school buses, pedestrians, railroads, ...) unless the description involves
  them. Copy the numbers exactly as written in the catalog.
- Use legal vocabulary in queries: say "traffic-control signal red indication", not "red light" (which is
  also the colour of emergency-vehicle lights). Do not use emergency-vehicle wording unless one is involved."""

ANALYZE_SYSTEM = """You are a traffic-law analyst assisting a human reviewer. You receive (1) facts of an
accident and (2) excerpts of law retrieved from a database.

Strict rules:
- Use ONLY the law excerpts provided. Never cite a section that is not in them.
- Cite every point as "VTL § <number>(<subdivision>)".
- If the excerpts do not cover a behavior, say so instead of inventing a rule.
- Do not assume facts that are not in the facts section. List what is missing.
- This is an analysis aid, not a legal judgment."""

ANALYZE_USER = """# ACCIDENT FACTS
{facts}

# RETRIEVED LAW EXCERPTS
{laws}

# TASK
Write the answer in {language} using exactly these sections:
1. **Facts used** (short bullets)
2. **Per vehicle**: for each vehicle, the possible violations, each with the section/subdivision and why it applies
3. **Most likely at fault**: who, and an approximate share of responsibility if it can be estimated; mention comparative fault if both drivers violated a rule
4. **Confidence**: high / medium / low, and the missing information that would change the conclusion
5. **Retrieved laws that do not apply** (one line each)"""


def build_catalog(parents):
    """'number | title' for every searchable section, so the LLM can pick sections by title."""
    def key(sec):
        m = re.match(r"\d+", sec)
        return (int(m.group()) if m else 0, sec)
    rows = sorted({(p["section"], p["title"]) for p in parents.values() if p["category"] == "core"},
                  key=lambda r: key(r[0]))
    return "\n".join(f"{sec} | {title[:110]}" for sec, title in rows)


def load_llm(model_name):
    print(f"Loading Hugging Face model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)

    return tokenizer, model


def call_llm(tokenizer, model, system, user, max_tokens=2000):

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in model output:\n{text[:500]}")
    return json.loads(text[start:end + 1])


def format_laws(laws):
    blocks = []
    for p in laws:
        words = p["text"].split()
        body = p["text"] if len(words) <= MAX_WORDS_PER_LAW else " ".join(words[:MAX_WORDS_PER_LAW]) + " [truncated]"
        blocks.append(f"## {p['id']} | matched by: {'; '.join(p.get('matched_queries', []))}\n{body}")
    return "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out")
    ap.add_argument("--file", help="text file with the accident description")
    ap.add_argument("--text", help="accident description given directly")
    ap.add_argument("--lang", default="en", choices=["en", "ar"], help="language of the final answer")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-k", type=int, default=8, help="number of law sections sent to the LLM")
    ap.add_argument("--no-title-routing", action="store_true",
                    help="do not let the LLM pick sections from the title catalog (vector search only)")
    ap.add_argument("--rerank", action="store_true", help="re-score retrieved chunks with a cross-encoder")
    ap.add_argument("--show-laws", action="store_true", help="print the full retrieved law text")
    ap.add_argument("--save", help="save the report to this .md file")
    a = ap.parse_args()

    if not (a.file or a.text):
        ap.error("give --file or --text")
    description = a.text or Path(a.file).read_text(encoding="utf-8")
    tokenizer, model = load_llm(a.model)

    parents, _ = load(a.out)

    # 1) facts + queries + sections picked by title
    print("[1/3] Extracting facts, search queries and relevant sections ...")
    if a.no_title_routing:
        user_msg = description
    else:
        user_msg = (f"SECTION CATALOG (number | title):\n{build_catalog(parents)}\n\n"
                    f"ACCIDENT DESCRIPTION:\n{description}")
    facts = parse_json(
    call_llm(
        tokenizer,
        model,
        EXTRACT_SYSTEM,
        user_msg
    )
)
    queries = facts.pop("queries", [])
    picked_raw = [] if a.no_title_routing else facts.pop("relevant_sections", [])
    facts.pop("relevant_sections", None)
    if not queries:
        raise SystemExit("The model returned no search queries.")
    print("Queries:", *[f"  - {q}" for q in queries], sep="\n")

    # sections the LLM picked from the catalog (only ones that really exist and are searchable)
    anchors, seen = [], set()
    for x in picked_raw:
        pid = f"VTL-{str(x).strip().lower().replace('section', '').replace('§', '').strip()}"
        if pid in parents and parents[pid]["category"] == "core" and pid not in seen:
            seen.add(pid)
            q = dict(parents[pid])
            q["matched_queries"] = ["picked by the LLM from the section titles"]
            q["score"], q["score_kind"] = None, "llm-pick"
            anchors.append(q)
    print("Picked by title:", [p["id"] for p in anchors] or "none")

    # 2) retrieval: LLM-picked sections first, vector search fills the remaining slots
    print("\n[2/3] Retrieving laws ...")
    n_anchor = min(len(anchors), max(1, a.top_k - 2))
    laws = anchors[:n_anchor]
    for p in retrieve(queries, a.out, top_k=a.top_k, expand_refs=True, rerank=a.rerank):
        if len(laws) >= a.top_k:
            break
        if p["id"] not in {x["id"] for x in laws}:
            laws.append(p)
    for p in laws:
        print(f"  {p['id']:<14} {p['title'][:50]:<50} score={p['score']}  <- {p['matched_queries']}")
    if a.show_laws:
        print("\n" + format_laws(laws))

    # 3) analysis
    print("\n[3/3] Analysing fault ...\n")
    language = "Arabic" if a.lang == "ar" else "English"
    answer = call_llm(
    tokenizer,
    model,
    ANALYZE_SYSTEM,
    ANALYZE_USER.format(
        facts=json.dumps(facts, ensure_ascii=False, indent=2),
        laws=format_laws(laws),
        language=language
    ),
    max_tokens=3000,
)
    print(answer)

    if a.save:
        report = (f"# Accident analysis\n\n## Description\n{description}\n\n## Extracted facts\n```json\n"
                  f"{json.dumps(facts, ensure_ascii=False, indent=2)}\n```\n\n## Queries\n"
                  + "\n".join(f"- {q}" for q in queries)
                  + f"\n\n## Retrieved sections\n" + ", ".join(p["id"] for p in laws)
                  + f"\n\n## Analysis\n{answer}\n")
        Path(a.save).write_text(report, encoding="utf-8")
        print(f"\nSaved to {a.save}")


if __name__ == "__main__":
    main()