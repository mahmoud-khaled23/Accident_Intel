import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv(_ROOT / '.env')

import os
import streamlit as st

st.title('🔧 Car Damage Cost Estimator')

st.info(
    'The Car Damage Cost Estimator is a self-contained Streamlit app. '
    'It uses YOLO damage detection and a cost-rules engine to produce '
    'an EGP repair estimate from vehicle photos.'
)

st.markdown("""
### What it does

1. **Upload photos** of the damaged vehicle (one or more angles).
2. **YOLO detection** segments damaged parts and identifies component types.
3. **Cost estimation** applies the configured cost rules to produce an itemised EGP estimate.
4. **Review & correct** the detections interactively before finalising the report.

### Prerequisites

- `GEMINI_API_KEY` set in `src/Car_damage_cost/.env` (for the VLM price-lookup step)
- YOLO weights downloaded to `src/Car_damage_cost/weights/`

Run the download script first if the weights are not yet present:

```bash
cd src/Car_damage_cost
python download_weights.py
```

### How to run

Launch the app in a separate terminal from the project root:
""")

_run_cmd = f'cd {_ROOT / "src" / "Car_damage_cost"} && streamlit run app.py'
st.code(_run_cmd, language='bash')

st.markdown("""
The app starts on **http://localhost:8501** by default, or choose a different port:
""")
st.code(
    'cd src/Car_damage_cost && streamlit run app.py --server.port 8502',
    language='bash',
)

st.markdown('### GEMINI_API_KEY status')
gemini_key = os.environ.get('GEMINI_API_KEY', '')
if gemini_key:
    st.success('✅ GEMINI_API_KEY is set in the environment.')
else:
    st.warning(
        '❌ GEMINI_API_KEY is not set. '
        'Add it to the project `.env` file or to `src/Car_damage_cost/.env`.'
    )

st.divider()
st.caption(
    'The damage cost module runs with `src/Car_damage_cost/` as its working directory '
    'because it uses relative imports. It cannot be embedded directly in this unified UI.'
)
