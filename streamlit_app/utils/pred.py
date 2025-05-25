# prediction.py 🔮 Forecasting and Inference Utilities

import joblib
import pandas as pd
from pathlib import Path

# --- Load Saved Model ---
def load_model(model_name, folder="data/preprocessed"):
    path = Path(folder) / model_name
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at: {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

# --- Make Prediction for a Single Input Row ---
def predict_single(model, input_dict):
    try:
        df = pd.DataFrame([input_dict])
        prediction = model.predict(df)[0]
        return prediction
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

# --- Make Prediction for Batch Data ---
def predict_batch(model, df):
    try:
        return model.predict(df)
    except Exception as e:
        raise RuntimeError(f"Batch prediction failed: {e}")

# --- Load Precomputed Forecast CSV (e.g., 2020–2050) ---
def load_forecast_csv(name: str, folder="data/preprocessed"):
    path = Path(folder) / name
    if not path.exists():
        raise FileNotFoundError(f"Forecast file not found: {path}")
    return pd.read_csv(path)

# --- Example: Load Climate Forecast ---
def get_climate_forecast():
    return load_forecast_csv("climate_forecast_2020_2050.csv")

# --- Example: Load Heatwave Forecast ---
def get_highheat_forecast():
    return load_forecast_csv("highheat_days_forecast_2020_2050.csv")

# --- Example: Load Drought Forecast ---
def get_drought_forecast():
    return load_forecast_csv("drought_forecast_spi_2020_2050.csv")

# --- Example: Load Glacier Forecast ---
def get_glacier_forecast():
    return load_forecast_csv("glacier_forecast_2020_2050.csv")
