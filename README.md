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

### 1. Univariate Forecasting Layer

- Stabilizes individual signals like:
    - Weather parameters (temperature, wind speed, irradiance)
    - Past energy generation
- Models used:
    - Holt-Winters
    - SARIMA
    - Prophet
    - N-BEATS

###  2. Multivariate Modeling Layer

- Combines:
    - Weather forecasts
    - Historical generation
    - Plant metadata
- Captures **non-linear relationships** between weather and power output

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

##  Web Application (Prototype)

A **Streamlit-based interactive dashboard** is built to demonstrate:

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
  
# Run app  
streamlit run app.py

---

## 📜 License

This project is currently for **hackathon and educational purposes**.  
Dataset is **synthetic**.
