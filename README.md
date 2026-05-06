# PRAVAAH

### AI-Powered Renewable Energy Forecasting & Decision Support System

> ⚡ _Turning weather uncertainty into actionable energy intelligence_

## Overview

Renewable energy sources like solar ☀️ and wind 🌬️ are inherently unpredictable due to their dependency on weather conditions. This creates challenges in power planning, grid stability, and operational efficiency.

**PRAVAAH** is an AI-driven forecasting and decision-support system designed to:

- Predict energy generation across **solar, wind, and hybrid plants**
- Provide **day-ahead and intra-day forecasts**
- Enable **data-driven decision-making for grid operators and planners**

This project is built as part of a **hackathon prototype**, focusing on scalability, explainability, and real-world applicability.

## Problem Statement

- Renewable generation is **highly volatile**
- Forecast errors lead to:
    - Grid imbalance 
    - Financial penalties 
    - Inefficient energy distribution 
 There is a need for a **robust, explainable, and scalable forecasting system**.

## Our Solution

PRAVAAH uses a **two-layer forecasting architecture**:

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

###  3. Decision Intelligence Layer

- Generates:
    - Final power forecasts 
    - Confidence intervals 
    - Key drivers of prediction 

##  System Architecture

The system is designed as a modular pipeline:

Data Sources → Data Processing → Univariate Models → Multivariate Model → Forecast Output → Dashboard

### Data Sources:

-  Weather API (real-time + forecast)
-  Historical generation data
-  Plant metadata

##  Dataset

We created a **realistic synthetic dataset** to simulate real-world conditions:

### Includes:

- 50 power plants:
    - Solar
    - Wind
    - Hybrid
- Hourly data:
    - Weather features
    - Power generation
- Forecast-ready structure

### Key Tables:

- `plant_master_data`
- `weather_data`
- `generation_data`

##  Project Structure

```bash
forecasting_pipeline/
│
├── data/
│   ├── forecasts/
│        ├── Availability, Irradiance, Health Factor, Curtailment, Generation_Univariate
│        ├── multivariate 
│           ├── Solar/Wind/Hybrid
│               ├── multivariate_forecast, model_selection_log, scenario_simulation
│           ├── stl_fleet_summary
├── models/
├── src/
│   ├── Availability, Irradiance, Health Factor, Curtailment, Generation_Univariate
│   ├── features.py
│   ├── preprocessing.py
│   ├── multivariate.py
├── pravaah.py
```

##  Web Application (Prototype)

A **Streamlit-based interactive dashboard** is built to demonstrate:
https://pravaah-renewable-energy.onrender.com/


### Features:

-  Live weather streaming
-  Historical + real-time + forecast visualization
-  Plant-level generation insights
-  Interactive filtering

### Users:

- Grid Operators
- Energy Analysts
- Plant Managers

##  Key Features

- Multi-plant forecasting (Solar + Wind + Hybrid)
-  Weather-integrated predictions
- Scalable architecture
- Explainable outputs
- Real-time + future insights

##  Tech Stack

- **Python** 
- **Pandas / NumPy**
- **Time Series Models**
- **Streamlit** (UI)
- **APIs** (Weather data)

##  Future Enhancements

-  Probabilistic forecasting
-  Advanced deep learning models
-  Grid optimization layer
-  Cloud deployment
-  Real-time API integration

## Team HuMachine 
1. Shuchismita Mallick           
2. Surya Vanshi Sah              

## AI FOR BHARAT Hackathon Submission

This project is developed as part of the **hackathon**, focusing on:

- Innovation 
- Practical feasibility 
- Real-world impact 

##  How to Run

# Clone repo  
git clone https://github.com/Shuchismita2000/pravah.git  
  
# Install dependencies  
pip install -r requirements.txt  

# Model Training 
run model_training.ipynb [unzip generation.csv]

# Run app  
streamlit run app.py

---

## 📜 License

This project is currently for **hackathon and educational purposes**.  
Dataset is **synthetic**.
