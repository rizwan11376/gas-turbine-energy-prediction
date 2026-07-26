"""Test the gas turbine /predict endpoint."""
import requests

# A realistic operating point (values near the training means).
sample = {
    "AT": 17.7,
    "AP": 1013.1,
    "AH": 77.9,
    "AFDP": 3.9,
    "GTEP": 25.6,
    "TIT": 1081.4,
    "TAT": 546.2,
    "CDP": 12.1,
}

response = requests.post("http://127.0.0.1:8000/predict", json=sample)
print("Status code:", response.status_code)
print("Response:", response.json())