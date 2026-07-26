# 🔧 Gas Turbine Energy Yield Prediction

Machine-learning regression system that predicts a combined-cycle gas turbine's hourly energy yield (MWh) from operational sensor readings. A tuned **HistGradientBoosting Regressor** is served through a **FastAPI** backend and an interactive **Streamlit** dashboard.

---

## Overview

Accurately forecasting a gas turbine's energy output from its operating conditions supports performance monitoring, efficiency analysis, and predictive maintenance in power plants. This project builds a regression model on five years of real hourly sensor data from a gas turbine in Turkey (2011–2015), with a strong emphasis on **honest, leakage-free evaluation**.

The defining methodological feature is **time-series (chronological) validation**: the model is trained on earlier years and tested on a genuinely later, unseen year — reflecting the real task of predicting future turbine behaviour rather than interpolating within shuffled data.

## The Problem

- **Task:** Regression — predict continuous turbine energy yield (TEY)
- **Dataset:** [Gas Turbine CO and NOx Emission Data Set](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set) (UCI, CC BY 4.0)
- **Size:** 36,733 hourly records across 2011–2015, 11 sensor measures
- **Target:** `TEY` — turbine energy yield (MWh), ranging 100–180

## Key Design Decisions

**Chronological validation, not random splitting.** This is time-series data. A random shuffle would let the model see "future" readings during training, inflating scores. Instead, an expanding-window `TimeSeriesSplit` trains on earlier years and validates on later ones, and the final test is the fully held-out **2015** data.

**Leakage prevention — CO and NOx excluded.** The dataset includes CO and NOx emission readings. These are combustion **by-products measured simultaneously with energy output**, not operational inputs a plant sets to control yield. Using them to predict TEY would be data leakage (predicting one output from another). They were dropped; the model uses only the 8 operational sensors.

**Feature set (8 operational sensors):**

| Code | Sensor | Role |
|---|---|---|
| CDP | Compressor discharge pressure | strongest predictor (r=0.99) |
| GTEP | Gas turbine exhaust pressure | strong (r=0.96) |
| TIT | Turbine inlet temperature | strong, non-linear (r=0.91) |
| TAT | Turbine after temperature | strong negative (r=-0.68) |
| AFDP | Air filter differential pressure | moderate |
| AT, AP, AH | Ambient temp / pressure / humidity | operational inputs |

## Results (2015 Test Set)

The tuned HistGradientBoosting model, evaluated **once** on the held-out 2015 data:

| Metric | Value |
|---|---|
| RMSE | **1.14 MWh** |
| MAE | 0.87 MWh |
| R² | **0.995** |

Context: the target's standard deviation is 15.62 MWh, so an RMSE of 1.14 is under 1.5% of the operating range. The model explains 99.5% of the variance in energy yield — a genuinely deterministic (thermodynamic) process, predicted accurately on an unseen future year with no sign of drift.

**Model comparison** (time-series cross-validated RMSE):

| Model | RMSE (MWh) | R² |
|---|---|---|
| HistGradientBoosting (selected) | 1.20 | 0.992 |
| Random Forest | 1.28 | 0.992 |
| Ridge Regression | 1.70 | 0.978 |
| Linear Regression | 1.72 | 0.977 |

The tree ensembles outperformed the linear models because the turbine-inlet-temperature relationship with yield is **non-linear** — linear models rode on the near-linear compressor-pressure signal but could not capture the curved TIT relationship.

## Known Limitation

The model slightly **under-predicts at peak energy yields above ~170 MWh**, where training examples are sparse (the turbine spends most time near ~134 MWh). High-load predictions should be treated with mild caution.

## Architecture

```
User enters 8 sensor values in Streamlit
            │
    (Streamlit loads the model directly)
            │
add_engineered → HistGradientBoosting pipeline
            │
   Predicted energy yield (MWh) shown on a gauge
```

The FastAPI backend (`api.py`) exposes the same model as a REST API for programmatic access; the Streamlit app is the interactive dashboard.

## Project Structure

```
gas-turbine-energy-prediction/
├── data/                     five yearly CSVs (2011-2015)
├── models/
│   └── model.joblib          saved regression pipeline
├── notebooks/
│   └── 01_gas_turbine_regression.ipynb   full EDA + modelling
├── src/
│   └── features.py           feature columns, labels, ranges (shared)
├── tests/
│   └── test_api.py           API test
├── api.py                    FastAPI backend
├── streamlit_app.py          Streamlit frontend
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/rizwan11376/gas-turbine-energy-prediction.git
cd gas-turbine-energy-prediction

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

## Running Locally

**Streamlit dashboard (loads the model directly):**
```bash
python -m streamlit run streamlit_app.py
# opens http://localhost:8501
```

**FastAPI backend (optional, for API access):**
```bash
python -m uvicorn api:app --reload
# serves http://127.0.0.1:8000  (docs at /redoc)
```

## API Reference

**POST /predict**
```json
{
  "AT": 17.7, "AP": 1013.1, "AH": 77.9, "AFDP": 3.9,
  "GTEP": 25.6, "TIT": 1081.4, "TAT": 546.2, "CDP": 12.1
}
```
Response:
```json
{ "predicted_tey_mwh": 135.03, "unit": "MWh" }
```

## Tech Stack

Python · scikit-learn · pandas · FastAPI · Streamlit · joblib

---

*A regression project demonstrating time-series validation, leakage-aware feature selection, model comparison, and full-stack deployment. Developed by Rizwan Ullah as a project for AI/ML training.*
