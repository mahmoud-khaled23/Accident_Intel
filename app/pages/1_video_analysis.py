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
    'vlm_video_name': None,
})

# ── Page ──────────────────────────────────────────────────────────────────────
st.title('🎥 Video Analysis (VLM)')

st.warning(
    '⚠️ This page requires a CUDA GPU and the Qwen2-VL model (~15 GB). '
    'First load takes several minutes.'
)


_IMPORT_FAILED = object()  # sentinel: deps missing, distinct from None or a bad return


@st.cache_resource(show_spinner='Loading Qwen2-VL model (~15 GB, unquantized)…')
def load_vlm_model(model_id: str):
    """Lazily import torch/transformers and load the Qwen2-VL model in native precision.

    Returns:
        (model, processor) tuple on success.
        _IMPORT_FAILED sentinel when required packages are not installed.
    Raises:
        Any non-ImportError exception from load_model() is re-raised so the
        caller can display it with st.error rather than silently returning None.
    """
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
    # Write uploaded bytes to a temp file
    tmp_dir = tempfile.mkdtemp(prefix='vlm_')
    tmp_video = str(Path(tmp_dir) / uploaded.name)
    with open(tmp_video, 'wb') as f:
        f.write(uploaded.read())

    # Load model in full precision
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

    # Build prompt
    try:
        from VLM.analyze_video import get_video_duration, build_prompt, analyze
        import types
        duration_s = get_video_duration(tmp_video)

        if prompt_mode == 'Custom' and custom_prompt:
            prompt = custom_prompt
        else:
            args_ns = types.SimpleNamespace(
                prompt_file=None,
                prompt_mode='json' if prompt_mode == 'JSON' else 'report',
                n_frames=n_frames,
            )
            prompt = build_prompt(args_ns, duration_s)

        with st.spinner('Analysing video…'):
            report = analyze(model, processor, tmp_video, prompt, n_frames, int(max_new_tokens))

        st.session_state['vlm_report'] = report
        st.session_state['vlm_video_name'] = uploaded.name

    except Exception as exc:
        st.error(f'Analysis failed: {exc}')

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.get('vlm_report'):
    report = st.session_state['vlm_report']
    video_name = st.session_state.get('vlm_video_name', 'video')

    st.divider()
    st.subheader('Analysis Report')
    st.markdown(report)

    st.download_button(
        label='Download report (.txt)',
        data=report,
        file_name=f'{Path(video_name).stem}_vlm_report.txt',
        mime='text/plain',
    )
