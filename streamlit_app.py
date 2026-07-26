"""Streamlit frontend for the Gas Turbine Energy Yield model."""
import math
import streamlit as st
import pandas as pd
from pathlib import Path
from joblib import load

from src.features import FEATURE_COLUMNS, FEATURE_LABELS, FEATURE_RANGES

# --- Page config ---------------------------------------------------------
st.set_page_config(
    page_title="Gas Turbine Energy Yield",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom styling (professional industrial theme) ----------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #0f1620 0%, #1a2634 100%);
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p, .stApp label, .stApp .stMarkdown { color: #e6edf3 !important; }

    /* SIDEBAR: dark background + readable light text */
    [data-testid="stSidebar"] { background: #10171f !important; }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }

    [data-testid="stMetricValue"] { color: #4fd1c5 !important; font-size: 2.2rem !important; }
    [data-testid="stMetricLabel"] { color: #9fb3c8 !important; }

    [data-testid="stNumberInput"] input {
        background-color: #1e2a3a !important;
        color: #e6edf3 !important;
        border: 1px solid #2d3f54 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0891b2, #0e7490) !important;
        color: white !important; border: none !important;
        font-weight: 600 !important; letter-spacing: 0.3px;
    }
    [data-testid="stExpander"] summary { color: #e6edf3 !important; }
    .metric-card {
        background: #1a2634; border: 1px solid #2d3f54; border-radius: 12px;
        padding: 16px 20px; margin-bottom: 8px; color: #e6edf3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load model once (cached) --------------------------------------------
@st.cache_resource
def load_model():
    path = Path(__file__).parent / "models" / "model.joblib"
    return load(path)

artifact = load_model()
pipeline = artifact["pipeline"]


# --- SVG helpers ---------------------------------------------------------
def turbine_icon(size: int = 90) -> str:
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="blade" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#4fd1c5"/>
          <stop offset="100%" stop-color="#0e7490"/>
        </linearGradient>
      </defs>
      <circle cx="50" cy="50" r="44" fill="none" stroke="#2d3f54" stroke-width="4"/>
      <circle cx="50" cy="50" r="8" fill="#9fb3c8"/>
      <g fill="url(#blade)">
        <ellipse cx="50" cy="24" rx="6" ry="18"/>
        <ellipse cx="50" cy="76" rx="6" ry="18"/>
        <ellipse cx="24" cy="50" rx="18" ry="6"/>
        <ellipse cx="76" cy="50" rx="18" ry="6"/>
        <ellipse cx="32" cy="32" rx="6" ry="16" transform="rotate(45 32 32)"/>
        <ellipse cx="68" cy="68" rx="6" ry="16" transform="rotate(45 68 68)"/>
        <ellipse cx="68" cy="32" rx="16" ry="6" transform="rotate(45 68 32)"/>
        <ellipse cx="32" cy="68" rx="16" ry="6" transform="rotate(45 32 68)"/>
      </g>
    </svg>
    """


def gauge_svg(value: float, vmin: float = 100, vmax: float = 180) -> str:
    frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    angle = math.pi * (1 - frac)
    cx, cy, r = 150, 150, 120
    x = cx + r * math.cos(angle)
    y = cy - r * math.sin(angle)
    if frac < 0.4:
        col = "#4fd1c5"
    elif frac < 0.75:
        col = "#f6c343"
    else:
        col = "#fb7185"
    return f"""
    <svg width="300" height="180" viewBox="0 0 300 180" xmlns="http://www.w3.org/2000/svg">
      <path d="M 30 150 A 120 120 0 0 1 270 150" fill="none"
            stroke="#2d3f54" stroke-width="16" stroke-linecap="round"/>
      <path d="M 30 150 A 120 120 0 0 1 {x:.1f} {y:.1f}" fill="none"
            stroke="{col}" stroke-width="16" stroke-linecap="round"/>
      <line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="4"/>
      <circle cx="{cx}" cy="{cy}" r="8" fill="{col}"/>
      <text x="30" y="172" fill="#9fb3c8" font-size="13" text-anchor="middle">{vmin:.0f}</text>
      <text x="270" y="172" fill="#9fb3c8" font-size="13" text-anchor="middle">{vmax:.0f}</text>
      <text x="150" y="110" fill="{col}" font-size="34" font-weight="bold" text-anchor="middle">{value:.1f}</text>
      <text x="150" y="132" fill="#9fb3c8" font-size="13" text-anchor="middle">MWh</text>
    </svg>
    """


# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.markdown(turbine_icon(64), unsafe_allow_html=True)
    st.markdown("### Gas Turbine Predictor")
    st.markdown(
        "<p style='font-size:0.85rem;'>ML-based energy yield estimation "
        "for combined-cycle gas turbines.</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("<p style='color:#4fd1c5; font-weight:700; margin-bottom:0;'>MODEL</p>"
                "<p style='font-size:0.85rem; margin-top:2px;'>HistGradient Boosting Regressor</p>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#4fd1c5; font-weight:700; margin-bottom:0;'>ACCURACY (2015 test)</p>"
                "<p style='font-size:0.85rem; margin-top:2px;'>R&sup2; 0.995 &nbsp;|&nbsp; RMSE 1.14 MWh</p>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#4fd1c5; font-weight:700; margin-bottom:0;'>VALIDATION</p>"
                "<p style='font-size:0.85rem; margin-top:2px;'>Time-series (chronological) on an unseen year</p>",
                unsafe_allow_html=True)
    st.divider()
    st.markdown(
        "<p style='font-size:0.8rem; color:#9fb3c8;'>Developed by</p>"
        "<p style='font-size:1.05rem; font-weight:700; margin-top:-10px;'>Rizwan Ullah</p>"
        "<p style='font-size:0.8rem; color:#9fb3c8; margin-top:-8px;'>A project for AI/ML Training</p>",
        unsafe_allow_html=True,
    )
    #st.caption("⚠️ Demonstration only — not certified monitoring equipment.")


# --- Header --------------------------------------------------------------
head_left, head_right = st.columns([1, 6])
with head_left:
    st.markdown(turbine_icon(90), unsafe_allow_html=True)
with head_right:
    st.title("Gas Turbine Energy Yield Predictor")
    st.markdown(
        "Estimate a combined-cycle gas turbine's hourly energy output (MWh) "
        "from live ambient and turbine sensor readings — a data-driven tool "
        "for power-plant performance monitoring."
    )

st.divider()

# --- Input panel ---------------------------------------------------------
st.subheader("Sensor Readings")
st.markdown(
    "<p style='color:#9fb3c8;'>Enter the current operating conditions. "
    "Defaults reflect typical plant operation.</p>",
    unsafe_allow_html=True,
)

inputs = {}
cols = st.columns(4)
for i, feature in enumerate(FEATURE_COLUMNS):
    low, high, default = FEATURE_RANGES[feature]
    with cols[i % 4]:
        inputs[feature] = st.number_input(
            FEATURE_LABELS[feature],
            min_value=float(low),
            max_value=float(high),
            value=float(default),
            step=0.1,
            help=f"Valid range: {low} to {high}",
        )
        st.markdown(
            f"<p style='color:#6b7f96; font-size:0.75rem; margin-top:-8px;'>"
            f"Range: {low} - {high}</p>",
            unsafe_allow_html=True,
        )

st.divider()

# --- Model details -------------------------------------------------------
with st.expander("🔬 Model & Methodology Details"):
    st.markdown(
        """
        **Algorithm:** HistGradient Boosting Regressor (scikit-learn) - a
        gradient-boosted tree ensemble that builds many shallow decision trees
        sequentially, each correcting the previous one's errors.

        **Why this model:** The relationship between turbine inlet temperature
        and energy yield is non-linear, so tree ensembles outperformed linear
        regression (R2 0.992 vs 0.977 in cross-validation).

        **Inputs (8 operational sensors):** ambient temperature, pressure, and
        humidity; air filter differential pressure; gas turbine exhaust
        pressure; turbine inlet and after temperatures; compressor discharge
        pressure. Emission readings (CO, NOx) were deliberately excluded as
        combustion by-products - using them would be data leakage.

        **Validation:** Time-series (chronological) cross-validation - trained
        on earlier years, tested on a later unseen year (2015).

        **Performance on the unseen 2015 data:**
        """
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", "1.14 MWh")
    m2.metric("MAE", "0.87 MWh")
    m3.metric("R2 Score", "0.995")
    st.caption(
        "Target ranges 100-180 MWh, so a 1.14 MWh error is under 1.5% of the "
        "operating range."
    )

# --- Prediction ----------------------------------------------------------
st.subheader("Predicted Energy Yield")

if st.button("⚡ Predict Energy Yield", type="primary", use_container_width=True):
    df_input = pd.DataFrame([inputs])[FEATURE_COLUMNS]
    predicted = float(pipeline.predict(df_input)[0])

    g1, g2 = st.columns([1, 1])
    with g1:
        st.markdown(gauge_svg(predicted), unsafe_allow_html=True)
    with g2:
        st.metric(label="Predicted Turbine Energy Yield", value=f"{predicted:.2f} MWh")
        if predicted < 120:
            note = "Low-load operation."
        elif predicted < 150:
            note = "Typical mid-range operation."
        else:
            note = "High-load operation - strong output."
        st.markdown(
            f"<div class='metric-card'>{note}<br>"
            f"<span style='color:#9fb3c8;'>Operating range: 100-180 MWh</span></div>",
            unsafe_allow_html=True,
        )
else:
    st.info("Set the sensor values above and click Predict to estimate energy yield.")