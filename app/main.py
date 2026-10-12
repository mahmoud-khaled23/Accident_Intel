import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv(_ROOT / '.env')

import os
import streamlit as st

st.set_page_config(
    page_title='Accident Intel',
    page_icon='🚗',
    layout='wide',
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('🚗 Accident Intel')
    st.caption('Evidence-first accident analysis')
    st.divider()

    st.markdown('**Navigation**')
    st.page_link('pages/1_video_analysis.py',    label='1 · Video Analysis (VLM)',   icon='🎥')
    st.page_link('pages/2_vehicle_tracking.py',  label='2 · Vehicle Tracking',       icon='🚦')
    st.page_link('pages/3_accident_analysis.py', label='3 · Accident Analysis',      icon='⚖️')
    st.page_link('pages/4_damage_cost.py',       label='4 · Damage Cost Estimator',  icon='🔧')

    st.divider()
    st.markdown('**API Keys**')
    groq_key = os.environ.get('GROQ_API_KEY', '')
    gemini_key = os.environ.get('GEMINI_API_KEY', '')
    st.markdown(f"{'✅' if groq_key else '❌'} GROQ_API_KEY")
    st.markdown(f"{'✅' if gemini_key else '❌'} GEMINI_API_KEY")

    st.divider()
    st.caption('⚠️ Analysis aid only — not legal advice.')

# ── Landing dashboard ─────────────────────────────────────────────────────────
st.title('🚗 Accident Intel')
st.markdown(
    'An integrated toolkit for traffic-accident evidence analysis: '
    'video understanding, vehicle tracking, law-based liability analysis, and damage cost estimation.'
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('### 🎥 Video Analysis')
    st.markdown(
        'Upload accident footage and let the Qwen2-VL vision-language model '
        'produce a detailed incident report from sampled frames.'
    )
    st.page_link('pages/1_video_analysis.py', label='Open →', icon='🎥')

with col2:
    st.markdown('### 🚦 Vehicle Tracking')
    st.markdown(
        'Run YOLO11x + BoT-SORT on your video to detect, track, and visualise '
        'individual vehicle trajectories across the full clip.'
    )
    st.page_link('pages/2_vehicle_tracking.py', label='Open →', icon='🚦')

with col3:
    st.markdown('### ⚖️ Accident Analysis')
    st.markdown(
        'Describe the accident in text; the RAG pipeline retrieves relevant '
        'traffic law sections and a Groq LLM produces a fault-liability analysis.'
    )
    st.page_link('pages/3_accident_analysis.py', label='Open →', icon='⚖️')

with col4:
    st.markdown('### 🔧 Damage Cost')
    st.markdown(
        'Upload vehicle photos to the standalone damage-cost estimator: '
        'YOLO segmentation + cost rules produce an EGP repair estimate.'
    )
    st.page_link('pages/4_damage_cost.py', label='Open →', icon='🔧')
