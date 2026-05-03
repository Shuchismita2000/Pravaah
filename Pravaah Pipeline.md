
#  1. High-Level Architecture

Think in **3 layers**:

### **(A) Data Layer**

- Sources:
    
    - Weather API (irradiance, temp, wind, etc.)
        
    - Historical plant data (curtailment, availability, health factor)
        
- Storage:
    
    - Raw → Data Lake (S3 / Blob / local)
        
    - Processed → Feature store (Parquet / DB)
        

---

### **(B) Feature + Forecast Layer**

You have **two stages**:

## Stage 1: Univariate Forecasts

Forecast individually:

- Irradiance (weather-driven)
    
- Curtailment
    
- Availability
    
- Health factor
    

 Models:

- SARIMAX / LightGBM / XGBoost / LSTM (depending on your notebook)
    

---

## Stage 2: Multivariate Forecast

Use:

- Forecasted values (from Stage 1)
    
- Weather inputs
    

Output:

- Final target (likely power generation or yield)
    

---

### **(C) Serving Layer**

- Pipeline runs every hour (or daily)
    
- Uses:
    
    - Latest weather forecast
        
    - Latest observed plant data
        
- Outputs:
    
    - Next **24h high-confidence forecast**
        
    - Optional: 72h extended forecast
        

---

#  2. Pipeline Flow (Important)

```
                ┌──────────────┐
                │ Weather API  │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │ Data Ingest  │
                └──────┬───────┘
                       │
                ┌──────▼─────────────┐
                │ Feature Engineering│
                └──────┬─────────────┘
                       │
     ┌─────────────────▼─────────────────┐
     │ Stage 1: Individual Forecasts     │
     │ (irradiance, health, etc.)        │
     └─────────────────┬─────────────────┘
                       │
                ┌──────▼─────────────┐
                │ Combine Features   │
                └──────┬─────────────┘
                       │
     ┌─────────────────▼─────────────────┐
     │ Stage 2: Multivariate Model       │
     └─────────────────┬─────────────────┘
                       │
                ┌──────▼─────────────┐
                │ 24h Forecast Output│
                └────────────────────┘
```

---

#  3. Key Design Decisions

###  Why 24h rolling forecast?

- Weather accuracy drops after 24h
    
- So:
    
    - Train for 72h
        
    - Serve only 24h (high confidence)
        

---

###  Strategy: Sliding Window Inference

Every run:

- Take latest data
    
- Predict next 24h
    
- Discard older predictions
    

---

#  4. Model Storage Strategy

use `.pkl` (plant wise) (good for now)

Store:

```bash
models/ 
  irradiance.pkl
  curtailment.pkl
  availability.pkl
  health.pkl
  final_multivariate.pkl
```

---

#  5. Code Structure (Modular)

##  Project Structure

```bash
forecasting_pipeline/
│
├── data/
├── models/
├── pipeline/
│   ├── ingest.py
│   ├── features.py
│   ├── stage1_forecast.py
│   ├── stage2_forecast.py
│   └── run_pipeline.py
│
├── utils/
│   └── model_loader.py
```

---

#  6. Core Code

##  Model Loader

```python
import joblib

def load_model(path):
    return joblib.load(path)
```

---

##  Stage 1 Forecast

```python
def stage1_forecast(df, models):
    results = {}

    for target in ['irradiance', 'curtailment', 'availability', 'health_factor']:
        model = models[target]
        features = df.drop(columns=[target], errors='ignore')

        preds = model.predict(features)
        results[target] = preds

    return results
```

---

##  Stage 2 (Multivariate)

```python
def stage2_forecast(stage1_outputs, weather_df, final_model):
    import pandas as pd

    df = pd.DataFrame(stage1_outputs)
    df = pd.concat([df, weather_df.reset_index(drop=True)], axis=1)

    preds = final_model.predict(df)
    return preds
```

---

## Main Pipeline

```python
def run_pipeline(new_weather_data, latest_data):
    from utils.model_loader import load_model

    # Load models
    models = {
        'irradiance': load_model('models/irradiance.pkl'),
        'curtailment': load_model('models/curtailment.pkl'),
        'availability': load_model('models/availability.pkl'),
        'health_factor': load_model('models/health.pkl'),
    }

    final_model = load_model('models/final_multivariate.pkl')

    # Merge latest + weather
    df = prepare_features(latest_data, new_weather_data)

    # Stage 1
    stage1_outputs = stage1_forecast(df, models)

    # Stage 2
    final_preds = stage2_forecast(stage1_outputs, new_weather_data, final_model)

    # Return only 24h
    return final_preds[:24]
```

---

#  7. Scheduling

Use:

- **Airflow / Prefect / Cron**
    

---
# 8. Improvements
###  Model Monitoring

Track:

- Drift in irradiance
    
- Error vs actual
    

---

### Retraining Strategy

- Weekly retraining
    
- Store versioned models
    

