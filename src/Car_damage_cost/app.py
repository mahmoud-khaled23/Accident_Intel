"""Streamlit UI.   Run:  streamlit run app.py

1. Photos     upload + run the detection pipeline, see the masks
2. Review     edit / delete / add damages (this is where needs_review rows get confirmed)
3. Cost       calculate the estimate from the reviewed table
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import Config
from cost_engine import estimate_cost, load_tables
from review_logic import (
    DAMAGE_TYPES,
    PART_CHOICES,
    SEVERITIES,
    fused_to_rows,
    make_df,
    rows_to_fused,
)
from schemas import CarInfo

st.set_page_config(page_title="Car damage cost estimator", layout="wide")

ACTION_LABEL = {
    "repair_only": "Repair",
    "paint_only": "Paint",
    "repair_and_paint": "Repair + paint",
    "replace": "Replace",
}


# ---------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_models():
    from vlm_client import VlmClient
    from yolo_wrapper import YoloSegWrapper

    cfg = Config.from_env()
    damage = YoloSegWrapper(cfg.damage_weights, conf=cfg.damage_conf, class_conf=cfg.class_conf)
    parts = YoloSegWrapper(cfg.parts_weights, conf=cfg.parts_conf, imgsz=cfg.parts_imgsz, iou=cfg.parts_iou)
    vlm = VlmClient(cfg.vlm_base_url, cfg.vlm_model, cfg.vlm_timeout, api=cfg.vlm_api, api_key=cfg.vlm_api_key, extra=cfg.vlm_extra)
    return cfg, damage, parts, vlm


@st.cache_resource
def load_price_agent():
    cfg = Config.from_env()
    if not cfg.gemini_api_key:
        return None
    from price_agent import PriceAgent

    return PriceAgent(cfg.gemini_api_key, cfg.gemini_model, cfg.price_cache_path)


# ---------------------------------------------------------------- state
ss = st.session_state
for k, v in {"report": None, "base_df": None, "debug_dir": None, "cost": None,
             "cost_sig": None, "cost_warnings": [], "ver": 0}.items():
    ss.setdefault(k, v)


def run_analysis(files, use_vlm: bool) -> None:
    from pipeline import analyze_case          # heavy imports only when needed

    cfg, damage, parts, vlm = load_models()
    tmp = Path(tempfile.mkdtemp(prefix="cda_"))
    paths = []
    for i, f in enumerate(files, 1):
        p = tmp / f"photo{i}{Path(f.name).suffix.lower()}"   # ascii name: cv2 on Windows hates unicode paths
        p.write_bytes(f.getbuffer())
        paths.append(p)

    with st.spinner("Detecting damage and car parts..."):
        report = analyze_case(paths, damage, parts, vlm if use_vlm else None, cfg, debug_dir=tmp / "debug")

    ss.report = report
    ss.debug_dir = tmp / "debug"
    ss.base_df = make_df(fused_to_rows(report.fused))
    ss.cost, ss.cost_sig, ss.cost_warnings = None, None, []
    ss.ver += 1                                  # new key => the editor starts from the new table


def fmt_range(lo: float, hi: float) -> str:
    return "-" if not hi else f"{lo:,.0f} – {hi:,.0f}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Car")
    make = st.text_input("Make", "Toyota")
    model = st.text_input("Model", "Corolla")
    year = st.number_input("Year", min_value=1980, max_value=2035, value=2018, step=1)
    tier = st.selectbox("Price tier", ["economy", "mid", "luxury"], index=1,
                        help="Sets labor and paint rates. Economy / mid / luxury.")
    st.header("Options")
    use_vlm = st.checkbox("Use VLM", value=True,
                          help="Fills in the part when the masks do not overlap. Slower.")
    use_web = st.checkbox("Look up replacement part prices online", value=True,
                          help="Only used for parts that must be replaced.")

# ---------------------------------------------------------------- header + upload
st.title("Car damage cost estimator")
files = st.file_uploader("Upload photos of the damaged car", type=["jpg", "jpeg", "png", "webp"],
                         accept_multiple_files=True)
if st.button("Analyze photos", type="primary", disabled=not files):
    run_analysis(files, use_vlm)

tab_photos, tab_review, tab_cost = st.tabs(["1. Photos", "2. Review damages", "3. Cost estimate"])

# ---------------------------------------------------------------- tab 1: photos
with tab_photos:
    report = ss.report
    if report is None:
        st.info("Upload photos and press **Analyze photos**.")
    else:
        for ph in report.photos:
            st.subheader(ph.photo_id)
            if not ph.usable:
                st.warning("Photo not used: " + "; ".join(ph.notes or ["unreadable"]))
                continue
            left, right = st.columns([3, 2])
            stem = Path(ph.photo_id).stem
            combined = Path(ss.debug_dir) / f"{stem}_combined.jpg"
            if combined.exists():
                left.image(str(combined), caption="Damage (bright) over car parts (faint). '??' = part unknown.")
                with left.expander("Show parts only / damage only"):
                    for kind in ("parts", "damage"):
                        p = Path(ss.debug_dir) / f"{stem}_{kind}.jpg"
                        if p.exists():
                            st.image(str(p), caption=kind)
            for n in ph.notes:
                right.warning(n)
            if ph.items:
                right.dataframe(
                    pd.DataFrame(
                        [{"damage": it.damage_type, "part": it.part, "conf": it.confidence,
                          "severity": it.severity.value, "part from": it.part_source} for it in ph.items]
                    ),
                    hide_index=True,
                )
            else:
                right.info("No damage detected in this photo.")
        for w in report.warnings:
            st.warning(w)

# ---------------------------------------------------------------- tab 2: review
edited = None
with tab_review:
    if ss.base_df is None:
        st.info("Analyze photos first.")
    else:
        st.markdown(
            "Check each row. **Change** the part / type / severity if the model got it wrong, "
            "**tick `reviewed`** when you are happy with a flagged row, **delete** wrong rows "
            "(select the row and press Delete), and **add a row** at the bottom for damage the models missed."
        )
        edited = st.data_editor(
            ss.base_df,
            key=f"editor_{ss.ver}",
            num_rows="dynamic",
            hide_index=True,
            disabled=["confidence", "needs_review"],
            column_config={
                "part": st.column_config.SelectboxColumn("Part", options=PART_CHOICES, required=True),
                "damage_type": st.column_config.SelectboxColumn("Damage", options=DAMAGE_TYPES, required=True),
                "severity": st.column_config.SelectboxColumn("Severity", options=SEVERITIES, required=True),
                "count": st.column_config.NumberColumn("Count", min_value=1, step=1, default=1),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                "needs_review": st.column_config.CheckboxColumn("Flagged", help="Flagged by the system as uncertain."),
                "reviewed": st.column_config.CheckboxColumn("Reviewed", help="Tick when you confirmed this row."),
            },
        )
        st.caption("Flagged rows you do not tick as reviewed are still counted, but shown as 'unconfirmed' in the cost.")

# ---------------------------------------------------------------- tab 3: cost
with tab_cost:
    if ss.report is None or edited is None:
        st.info("Analyze photos and review the damages first.")
    else:
        cfg = Config.from_env()
        sig = f"{make}|{model}|{year}|{tier}|{use_web}|{edited.to_json()}"

        if st.button("Calculate cost", type="primary", key="calc"):
            fused, warns = rows_to_fused(edited.to_dict("records"))
            if not make.strip() or not model.strip():
                st.error("Enter the car make and model in the sidebar.")
            else:
                car = CarInfo(make=make.strip(), model=model.strip(), year=int(year), tier=tier)
                agent = load_price_agent() if use_web else None
                if use_web and agent is None:
                    st.warning("GEMINI_API_KEY is not set: using the placeholder price table.")
                with st.spinner("Calculating (looking up part prices can take a minute)..."):
                    ss.cost = estimate_cost(fused, car, load_tables(cfg.prices_path), agent)
                ss.cost_sig, ss.cost_warnings = sig, warns

        cost = ss.cost
        if cost is not None:
            if ss.cost_sig != sig:
                st.warning("The damages or car details changed since this estimate. Press **Calculate cost** again.")
            for w in ss.cost_warnings:
                st.warning(w)

            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated total (EGP)", fmt_range(cost.total_min, cost.total_max))
            c2.metric("Of which unconfirmed", fmt_range(cost.unconfirmed_min, cost.unconfirmed_max))
            c3.metric("Parts to work on", len(cost.lines))

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Part": l.part,
                            "Work": ACTION_LABEL.get(l.action, l.action),
                            "Damage": ", ".join(l.damages),
                            "New part": fmt_range(l.parts_min, l.parts_max),
                            "Labor": fmt_range(l.labor_min, l.labor_max),
                            "Paint": fmt_range(l.paint_min, l.paint_max),
                            "Total (EGP)": fmt_range(l.total_min, l.total_max),
                            "Price source": l.price_source,
                            "Unconfirmed": "yes" if l.needs_review else "",
                        }
                        for l in cost.lines
                    ]
                ),
                hide_index=True,
            )

            for l in cost.lines:
                for n in l.notes:
                    st.caption(f"{l.part}: {n}")
            for u in cost.unpriced:
                st.warning(f"Not priced (unknown part): {u}. Pick the part in the Review tab.")
            for n in cost.notes:
                st.caption(n)

            st.download_button("Download cost report (JSON)", cost.model_dump_json(indent=2),
                               file_name="cost.json", mime="application/json")