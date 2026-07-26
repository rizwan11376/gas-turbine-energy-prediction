"""FastAPI backend for the gas turbine energy yield model."""
from pathlib import Path
import pandas as pd
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import FEATURE_COLUMNS

# --- Load the model once at startup --------------------------------------
MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"
artifact = load(MODEL_PATH)
pipeline = artifact["pipeline"]

app = FastAPI(title="Gas Turbine Energy Yield Prediction API")


# --- Input schema: the 8 operational sensors -----------------------------
class TurbineInput(BaseModel):
    AT:   float = Field(..., description="Ambient temperature (°C)")
    AP:   float = Field(..., description="Ambient pressure (mbar)")
    AH:   float = Field(..., description="Ambient humidity (%)")
    AFDP: float = Field(..., description="Air filter diff. pressure (mbar)")
    GTEP: float = Field(..., description="Gas turbine exhaust pressure (mbar)")
    TIT:  float = Field(..., description="Turbine inlet temperature (°C)")
    TAT:  float = Field(..., description="Turbine after temperature (°C)")
    CDP:  float = Field(..., description="Compressor discharge pressure (mbar)")


@app.get("/")
def root():
    return {"message": "Gas Turbine Energy Yield Prediction API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict(data: TurbineInput):
    """Predict turbine energy yield (TEY) from 8 operational sensors."""
    # Build a one-row DataFrame in the exact column order the model expects.
    row = {col: getattr(data, col) for col in FEATURE_COLUMNS}
    df_input = pd.DataFrame([row])[FEATURE_COLUMNS]

    # Regression: predict returns the number directly. No threshold.
    predicted_tey = float(pipeline.predict(df_input)[0])

    return {
        "predicted_tey_mwh": round(predicted_tey, 2),
        "unit": "MWh",
    }