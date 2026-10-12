import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv(_ROOT / '.env')

import json
import os
import streamlit as st

from app.utils.session import init_session_defaults

# ── RAG index path (has chroma/ subdirectory) ─────────────────────────────────
_RAG_OUT = str(_ROOT / 'src' / 'rag' / 'chunking' / 'out')

# ── Session state defaults ────────────────────────────────────────────────────
init_session_defaults({
    'accident_facts': None,
    'accident_laws': None,
    'accident_analysis': None,
    'accident_description': None,
})

# ── Page ──────────────────────────────────────────────────────────────────────
st.title('⚖️ Accident Analysis')

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Settings')
    groq_api_key = st.text_input(
        'GROQ API Key',
        value=os.environ.get('GROQ_API_KEY', ''),
        type='password',
        label_visibility='visible',
    )
    st.caption('[Get a free key at console.groq.com](https://console.groq.com)')

    with st.expander('Advanced options', expanded=False):
        model_id = st.text_input('Model', value='openai/gpt-oss-120b')
        top_k = st.slider('Retrieved law sections (top_k)', min_value=3, max_value=15, value=8)
        show_laws = st.checkbox('Show retrieved law sections', value=False)

    language = st.radio('Output language', ['English', 'Arabic'])

# ── Input mode ────────────────────────────────────────────────────────────────
input_mode = st.radio('Input mode', ['Text', 'File upload'], horizontal=True)

description = ''
if input_mode == 'Text':
    description = st.text_area(
        'Accident description',
        value=st.session_state.get('accident_description') or '',
        height=200,
        placeholder='Describe the accident in detail…',
    )
else:
    txt_file = st.file_uploader('Upload accident description (.txt)', type=['txt'])
    if txt_file is not None:
        description = txt_file.read().decode('utf-8')
        st.text_area('Loaded description', value=description, height=150, disabled=True)

# ── Analyse button ────────────────────────────────────────────────────────────
can_analyse = bool(description.strip()) and bool(groq_api_key.strip())
if st.button('Analyze', disabled=not can_analyse, type='primary'):
    st.session_state['accident_description'] = description

    try:
        from rag.retrieval.analyze_accident import (
            build_catalog,
            _call_groq,
            parse_json,
            format_laws,
            EXTRACT_SYSTEM,
            ANALYZE_SYSTEM,
            ANALYZE_USER,
        )
        from rag.retrieval.index_and_retrieve import load, retrieve
    except ImportError as e:
        st.error(
            f'RAG dependencies are not installed ({e}). Run:\n\n'
            '```\npip install chromadb sentence-transformers rank-bm25 openai\n```'
        )
        st.stop()

    # Step 1 — load index
    with st.spinner('Loading RAG index…'):
        try:
            parents, _ = load(_RAG_OUT)
        except Exception as exc:
            st.error(f'Failed to load RAG index at `{_RAG_OUT}`: {exc}')
            st.stop()

    # Step 2 — fact extraction
    with st.spinner('Step 1/3: Extracting facts and queries…'):
        try:
            catalog = build_catalog(parents)
            user_msg = (
                f'SECTION CATALOG (number | title):\n{catalog}\n\n'
                f'ACCIDENT DESCRIPTION:\n{description}'
            )
            facts_raw = _call_groq(
                EXTRACT_SYSTEM,
                user_msg,
                model=model_id,
                api_key=groq_api_key,
                max_tokens=1500,
            )
            facts = parse_json(facts_raw)
        except Exception as exc:
            st.error(f'Fact extraction failed: {exc}')
            st.stop()

    queries = facts.pop('queries', [])
    picked_raw = facts.pop('relevant_sections', [])
    facts.pop('relevant_sections', None)

    st.session_state['accident_facts'] = facts

    # Step 3 — retrieval
    with st.spinner('Step 2/3: Retrieving relevant law sections…'):
        try:
            anchors, seen = [], set()
            for x in picked_raw:
                pid = f"VTL-{str(x).strip().lower()}"
                if pid in parents and parents[pid]['category'] == 'core' and pid not in seen:
                    seen.add(pid)
                    q = dict(parents[pid])
                    q['matched_queries'] = ['picked by LLM from section titles']
                    q['score'], q['score_kind'] = None, 'llm-pick'
                    anchors.append(q)

            laws = anchors[:max(1, top_k - 2)]
            for p in retrieve(queries, _RAG_OUT, top_k=top_k, expand_refs=True, rerank=False):
                if len(laws) >= top_k:
                    break
                if p['id'] not in {x['id'] for x in laws}:
                    laws.append(p)
        except Exception as exc:
            st.error(f'Retrieval failed: {exc}')
            st.stop()

    st.session_state['accident_laws'] = laws

    # Step 4 — analysis
    with st.spinner('Step 3/3: Generating liability analysis…'):
        try:
            lang_label = 'Arabic' if language == 'Arabic' else 'English'
            analysis = _call_groq(
                ANALYZE_SYSTEM,
                ANALYZE_USER.format(
                    facts=json.dumps(facts, ensure_ascii=False, indent=2),
                    laws=format_laws(laws),
                    language=lang_label,
                ),
                model=model_id,
                api_key=groq_api_key,
                max_tokens=3000,
            )
        except Exception as exc:
            st.error(f'Analysis failed: {exc}')
            st.stop()

    st.session_state['accident_analysis'] = analysis

# ── Results ───────────────────────────────────────────────────────────────────
facts = st.session_state.get('accident_facts')
laws = st.session_state.get('accident_laws')
analysis = st.session_state.get('accident_analysis')

if facts is not None:
    with st.expander('Extracted facts (JSON)', expanded=False):
        st.json(facts)

if laws and show_laws:
    with st.expander(f'Retrieved law sections ({len(laws)})', expanded=False):
        for law in laws:
            st.markdown(f"**{law['id']}** — {law.get('title', '')}")
            st.caption(f"Matched by: {'; '.join(law.get('matched_queries', []))}")

if analysis:
    st.divider()
    st.subheader('Analysis')
    st.markdown(analysis)

    # Build downloadable report
    desc = st.session_state.get('accident_description', '')
    facts_md = json.dumps(facts or {}, ensure_ascii=False, indent=2)
    queries_section = ''
    law_ids = ', '.join(p['id'] for p in (laws or []))
    report_md = (
        f'# Accident Analysis\n\n'
        f'## Description\n{desc}\n\n'
        f'## Extracted Facts\n```json\n{facts_md}\n```\n\n'
        f'## Retrieved Law Sections\n{law_ids}\n\n'
        f'## Analysis\n{analysis}\n'
    )

    st.download_button(
        label='Download full report (.md)',
        data=report_md,
        file_name='accident_analysis.md',
        mime='text/markdown',
    )
