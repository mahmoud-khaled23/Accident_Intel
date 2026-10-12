import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv(_ROOT / '.env')

import json
import tempfile
import streamlit as st

from app.utils.session import init_session_defaults

# ── Session state defaults ────────────────────────────────────────────────────
init_session_defaults({
    'tracking_done': False,
    'tracking_output_video': None,
    'tracking_output_json': None,
    'tracking_trajectories': None,
})

# ── Page ──────────────────────────────────────────────────────────────────────
st.title('🚦 Vehicle Tracking')

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    'Upload video',
    type=['mp4', 'avi', 'mov', 'mkv'],
)

# ── Controls ──────────────────────────────────────────────────────────────────
model_path = st.text_input('YOLO weights', value='yolo11x.pt')

# ── Track button ──────────────────────────────────────────────────────────────
can_track = uploaded is not None
if st.button('Track vehicles', disabled=not can_track, type='primary'):
    tmp_dir = tempfile.mkdtemp(prefix='trk_')
    tmp_video = str(Path(tmp_dir) / uploaded.name)
    with open(tmp_video, 'wb') as f:
        f.write(uploaded.read())

    output_video = str(Path(tmp_dir) / 'tracked.mp4')
    output_json = str(Path(tmp_dir) / 'trajectories.json')

    try:
        from vehicle_tracking.tracker import process_video

        with st.spinner('Tracking vehicles…'):
            trajectories = process_video(
                input_path=tmp_video,
                output_video=output_video,
                output_json=output_json,
                model=None,  # always None so BoT-SORT resets per video
            )

        st.session_state['tracking_done'] = True
        st.session_state['tracking_output_video'] = output_video
        st.session_state['tracking_output_json'] = output_json
        st.session_state['tracking_trajectories'] = trajectories

    except ImportError as e:
        st.error(
            f'Tracking dependencies are not installed ({e}). Run:\n\n'
            '```\npip install ultralytics supervision\n```'
        )
    except Exception as exc:
        st.error(f'Tracking failed: {exc}')

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.get('tracking_done') and st.session_state.get('tracking_output_video'):
    output_video = st.session_state['tracking_output_video']
    output_json = st.session_state['tracking_output_json']
    trajectories = st.session_state['tracking_trajectories'] or {}

    st.divider()
    st.subheader('Results')

    # Video playback + download
    if Path(output_video).is_file():
        st.video(output_video)
        with open(output_video, 'rb') as vf:
            st.download_button(
                label='Download annotated video (.mp4)',
                data=vf,
                file_name='tracked.mp4',
                mime='video/mp4',
            )

    # JSON download
    if output_json and Path(output_json).is_file():
        with open(output_json, 'r', encoding='utf-8') as jf:
            json_str = jf.read()
        st.download_button(
            label='Download trajectories (.json)',
            data=json_str,
            file_name='trajectories.json',
            mime='application/json',
        )

    # Summary table
    if trajectories:
        st.subheader('Track Summary')
        rows = []
        for track_id, info in trajectories.items():
            frames_detected = len(info.get('trajectory', {}))
            rows.append({
                'track_id': track_id,
                'class': info.get('class', 'unknown'),
                'frames_detected': frames_detected,
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Trajectory visualisation
    if trajectories:
        st.subheader('Trajectories')

        try:
            import plotly.graph_objects as go  # noqa: F401
            _has_plotly = True
        except ImportError:
            _has_plotly = False

        if _has_plotly:
            import plotly.graph_objects as go
            fig = go.Figure()
            for track_id, info in trajectories.items():
                traj = info.get('trajectory', {})
                if not traj:
                    continue
                sorted_frames = sorted(traj.keys(), key=lambda k: int(k))
                cx_vals = [traj[f]['center'][0] for f in sorted_frames]
                cy_vals = [traj[f]['center'][1] for f in sorted_frames]
                frame_nums = [int(f) for f in sorted_frames]
                fig.add_trace(go.Scatter(
                    x=frame_nums,
                    y=cx_vals,
                    mode='lines',
                    name=f'ID {track_id} ({info.get("class","?")}) cx',
                ))
                fig.add_trace(go.Scatter(
                    x=frame_nums,
                    y=cy_vals,
                    mode='lines',
                    name=f'ID {track_id} ({info.get("class","?")}) cy',
                    line=dict(dash='dash'),
                ))
            fig.update_layout(
                xaxis_title='Frame',
                yaxis_title='Pixel coordinate',
                legend_title='Track / axis',
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Fallback: one line_chart per track
            for track_id, info in trajectories.items():
                traj = info.get('trajectory', {})
                if not traj:
                    continue
                sorted_frames = sorted(traj.keys(), key=lambda k: int(k))
                data = {
                    'frame': [int(f) for f in sorted_frames],
                    'cx': [traj[f]['center'][0] for f in sorted_frames],
                    'cy': [traj[f]['center'][1] for f in sorted_frames],
                }
                import pandas as pd
                df = pd.DataFrame(data).set_index('frame')
                st.markdown(f'**Track ID {track_id}** — {info.get("class", "?")}')
                st.line_chart(df)
