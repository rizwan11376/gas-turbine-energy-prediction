"""Shared definitions for the gas turbine energy yield model.

Single source of truth for the feature columns (the model's input contract)
and human-readable labels used in the GUI and plots.
"""

# The 8 operational sensors the model expects, in exact order.
# CO and NOX are excluded (co-product emissions = leakage).
FEATURE_COLUMNS = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP"]

TARGET = "TEY"

# Human-readable labels — DISPLAY ONLY (GUI, plots, docs).
# The model always uses the short codes above.
FEATURE_LABELS = {
    "AT":   "Ambient Temperature (°C)",
    "AP":   "Ambient Pressure (mbar)",
    "AH":   "Ambient Humidity (%)",
    "AFDP": "Air Filter Diff. Pressure (mbar)",
    "GTEP": "Gas Turbine Exhaust Pressure (mbar)",
    "TIT":  "Turbine Inlet Temperature (°C)",
    "TAT":  "Turbine After Temperature (°C)",
    "CDP":  "Compressor Discharge Pressure (mbar)",
    "TEY":  "Turbine Energy Yield (MWh)",
}

# Realistic input ranges for GUI widgets, from the training data.
# (min, max, typical_default) per sensor — we'll fill these from your data.
# Realistic input ranges for GUI widgets, from the training data.
# (min, max, default) per sensor — default is the training mean.
FEATURE_RANGES = {
    "AT":   (-7.0, 38.0, 17.7),
    "AP":   (985.0, 1037.0, 1013.1),
    "AH":   (24.0, 101.0, 77.9),
    "AFDP": (2.0, 8.0, 3.9),
    "GTEP": (17.0, 41.0, 25.6),
    "TIT":  (1000.0, 1101.0, 1081.4),
    "TAT":  (510.0, 551.0, 546.2),
    "CDP":  (9.8, 15.5, 12.1),
}