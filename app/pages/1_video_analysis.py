import os
import sys
from pathlib import Path

# Prevent Transformers async weight-loading deadlock on cloud/NFS file systems (Lightning AI)
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'src' / 'VLM'))

from dotenv import load_dotenv
load_dotenv(_ROOT / '.env')

# Fix PyAV >= 19 compatibility with torchvision.io.read_video
try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        kwargs.pop("metadata_encoding", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open
except (ImportError, AttributeError):
    pass

import tempfile
import streamlit as st


from app.utils.session import init_session_defaults

# ── Session state defaults ────────────────────────────────────────────────────
init_session_defaults({
    'vlm_report': None,
    'vlm_report_pure': None,
    'vlm_report_tracked': None,
    'vlm_tracking_summary': None,
    'vlm_tracked_video': None,
    'vlm_video_name': None,
    'vlm_mode': None,
})

# ── Page ──────────────────────────────────────────────────────────────────────
st.title('🎥 Video Analysis (VLM)')

st.info(
    '💡 **Two Paradigms Available:**\n'
    '- **Paradigm 1 (Pure VLM):** End-to-end multimodal perception on raw footage.\n'
    '- **Paradigm 2 (Grounded VLM):** Two-stage neuro-symbolic pipeline using YOLO11 + BoT-SORT to eliminate hallucinations.'
)


_IMPORT_FAILED = object()  # sentinel: deps missing, distinct from None or a bad return


@st.cache_resource(show_spinner='Loading Qwen2-VL model (~15 GB, unquantized)…')
def load_vlm_model(model_id: str):
    """Lazily import torch/transformers and load the Qwen2-VL model in native precision."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import qwen_vl_utils  # noqa: F401
        from VLM.analyze_video import load_model
    except ImportError:
        return _IMPORT_FAILED
    return load_model(model_id)


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    'Upload accident video',
    type=['mp4', 'avi', 'mov', 'mkv'],
)

# ── Controls ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    paradigm_choice = st.radio(
        'Paradigm Mode',
        [
            'Paradigm 1: Pure VLM (Without Tracking)',
            'Paradigm 2: Grounded VLM (With Detector & Tracker)',
            'Dual Comparison (Run Both & Compare)',
        ],
        index=1,
    )
    n_frames = st.slider('Sampled frames', min_value=8, max_value=64, value=32, step=4)
    prompt_mode = st.radio('Prompt mode', ['Report', 'JSON', 'Custom'])
    model_id = st.text_input('Model ID', value='Qwen/Qwen2-VL-7B-Instruct')
    max_new_tokens = st.number_input('Max new tokens', min_value=100, max_value=4000, value=700)

custom_prompt = None
if prompt_mode == 'Custom':
    custom_prompt = st.text_area(
        'Custom prompt',
        height=150,
        placeholder='Enter your custom prompt here…',
    )

# ── Analyse button ────────────────────────────────────────────────────────────
can_analyse = uploaded is not None
if st.button('Analyze video', disabled=not can_analyse, type='primary'):
    tmp_dir = tempfile.mkdtemp(prefix='vlm_')
    tmp_video = str(Path(tmp_dir) / uploaded.name)
    with open(tmp_video, 'wb') as f:
        f.write(uploaded.read())

    # Load model
    result = load_vlm_model(model_id)
    if result is _IMPORT_FAILED:
        st.error(
            'VLM dependencies are not installed. Run:\n\n'
            '```\npip install torch torchvision transformers accelerate '
            'qwen-vl-utils\n```\n\n'
            'A CUDA GPU is required for reasonable performance.'
        )
        st.stop()

    model, processor = result

    from VLM.analyze_video import get_video_duration, build_prompt, analyze
    import types
    duration_s = get_video_duration(tmp_video)

    args_ns = types.SimpleNamespace(
        prompt_file=None,
        prompt_mode='json' if prompt_mode == 'JSON' else 'report',
        n_frames=n_frames,
    )

    try:
        # ── Mode 1: Pure VLM (Without Tracking) ───────────────────────────────
        if paradigm_choice == 'Paradigm 1: Pure VLM (Without Tracking)':
            prompt = custom_prompt if (prompt_mode == 'Custom' and custom_prompt) else build_prompt(args_ns, duration_s, tracking_summary=None)
            with st.spinner('Running Paradigm 1: Pure VLM analysis on raw video…'):
                report = analyze(model, processor, tmp_video, prompt, n_frames, int(max_new_tokens))
            st.session_state['vlm_report'] = report
            st.session_state['vlm_mode'] = 'pure'
            st.session_state['vlm_tracking_summary'] = None
            st.session_state['vlm_tracked_video'] = None

        # ── Mode 2: Grounded VLM (With Tracking) ─────────────────────────────
        elif paradigm_choice == 'Paradigm 2: Grounded VLM (With Detector & Tracker)':
            from vehicle_tracking.tracker import process_video
            from vehicle_tracking.accident_candidates import format_tracking_summary, identify_accident_candidates
            import cv2

            cap = cv2.VideoCapture(tmp_video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            tracked_video = str(Path(tmp_dir) / 'tracked.mp4')
            tracked_json = str(Path(tmp_dir) / 'trajectories.json')

            with st.spinner('Stage 1/2: Running YOLO + BoT-SORT detector & tracker…'):
                trajectories = process_video(
                    input_path=tmp_video,
                    output_video=tracked_video,
                    output_json=tracked_json,
                )
                candidates = identify_accident_candidates(trajectories)
                tracking_summary = format_tracking_summary(trajectories, fps=fps, candidates=candidates)

            prompt = custom_prompt if (prompt_mode == 'Custom' and custom_prompt) else build_prompt(args_ns, duration_s, tracking_summary=tracking_summary)
            with st.spinner('Stage 2/2: Running Grounded VLM inference…'):
                report = analyze(model, processor, tracked_video, prompt, n_frames, int(max_new_tokens))

            st.session_state['vlm_report'] = report
            st.session_state['vlm_mode'] = 'tracked'
            st.session_state['vlm_tracking_summary'] = tracking_summary
            st.session_state['vlm_tracked_video'] = tracked_video

        # ── Mode 3: Dual Comparison (Run Both) ────────────────────────────────
        else:
            from vehicle_tracking.tracker import process_video
            from vehicle_tracking.accident_candidates import format_tracking_summary, identify_accident_candidates
            import cv2

            cap = cv2.VideoCapture(tmp_video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            tracked_video = str(Path(tmp_dir) / 'tracked.mp4')
            tracked_json = str(Path(tmp_dir) / 'trajectories.json')

            # 1. Tracker run
            with st.spinner('[Benchmark 1/3] Running YOLO + BoT-SORT tracking…'):
                trajectories = process_video(
                    input_path=tmp_video,
                    output_video=tracked_video,
                    output_json=tracked_json,
                )
                candidates = identify_accident_candidates(trajectories)
                tracking_summary = format_tracking_summary(trajectories, fps=fps, candidates=candidates)

            # 2. Paradigm 1: Pure VLM
            prompt_pure = custom_prompt if (prompt_mode == 'Custom' and custom_prompt) else build_prompt(args_ns, duration_s, tracking_summary=None)
            with st.spinner('[Benchmark 2/3] Running Paradigm 1: Pure VLM on raw footage…'):
                report_pure = analyze(model, processor, tmp_video, prompt_pure, n_frames, int(max_new_tokens))

            # 3. Paradigm 2: Grounded VLM
            prompt_tracked = custom_prompt if (prompt_mode == 'Custom' and custom_prompt) else build_prompt(args_ns, duration_s, tracking_summary=tracking_summary)
            with st.spinner('[Benchmark 3/3] Running Paradigm 2: Grounded VLM with telemetry…'):
                report_tracked = analyze(model, processor, tracked_video, prompt_tracked, n_frames, int(max_new_tokens))

            st.session_state['vlm_report_pure'] = report_pure
            st.session_state['vlm_report_tracked'] = report_tracked
            st.session_state['vlm_mode'] = 'compare'
            st.session_state['vlm_tracking_summary'] = tracking_summary
            st.session_state['vlm_tracked_video'] = tracked_video

        st.session_state['vlm_video_name'] = uploaded.name

    except Exception as exc:
        st.error(f'Analysis failed: {exc}')


# ── Results Display ───────────────────────────────────────────────────────────
vlm_mode = st.session_state.get('vlm_mode')
video_name = st.session_state.get('vlm_video_name', 'video')
stem = Path(video_name).stem

if vlm_mode == 'compare':
    st.divider()
    st.subheader('⚖️ Dual-Paradigm Benchmark Results')

    tab_compare, tab_pure, tab_tracked, tab_telemetry = st.tabs([
        'Side-by-Side Comparison',
        'Paradigm 1 (Without Tracking)',
        'Paradigm 2 (With Tracking)',
        'Tracking Telemetry & Video',
    ])

    with tab_compare:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### 🔵 Paradigm 1: Pure VLM')
            st.caption('Direct visual perception without tracker grounding')
            st.markdown(st.session_state.get('vlm_report_pure', ''))
        with col2:
            st.markdown('### 🟢 Paradigm 2: Grounded VLM')
            st.caption('Grounded with YOLO detection & BoT-SORT telemetry')
            st.markdown(st.session_state.get('vlm_report_tracked', ''))

    with tab_pure:
        st.markdown(st.session_state.get('vlm_report_pure', ''))
        st.download_button(
            label='Download Paradigm 1 Report (.txt)',
            data=st.session_state.get('vlm_report_pure', ''),
            file_name=f'{stem}_paradigm1_pure.txt',
            mime='text/plain',
        )

    with tab_tracked:
        st.markdown(st.session_state.get('vlm_report_tracked', ''))
        st.download_button(
            label='Download Paradigm 2 Report (.txt)',
            data=st.session_state.get('vlm_report_tracked', ''),
            file_name=f'{stem}_paradigm2_grounded.txt',
            mime='text/plain',
        )

    with tab_telemetry:
        if st.session_state.get('vlm_tracked_video') and Path(st.session_state['vlm_tracked_video']).is_file():
            st.video(st.session_state['vlm_tracked_video'])
        if st.session_state.get('vlm_tracking_summary'):
            st.text_area('Extracted Ground Truth Telemetry', value=st.session_state['vlm_tracking_summary'], height=250)

elif st.session_state.get('vlm_report'):
    report = st.session_state['vlm_report']
    st.divider()
    st.subheader(f"Analysis Report ({'Grounded with Tracking' if vlm_mode == 'tracked' else 'Without Tracking'})")
    st.markdown(report)

    if st.session_state.get('vlm_tracked_video') and Path(st.session_state['vlm_tracked_video']).is_file():
        st.subheader('Tracked Video (Bounding Boxes & IDs)')
        st.video(st.session_state['vlm_tracked_video'])

    st.download_button(
        label='Download report (.txt)',
        data=report,
        file_name=f'{stem}_vlm_report.txt',
        mime='text/plain',
    )

