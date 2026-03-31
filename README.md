# Multi-Domain Demand Forecasting

**Live Demo:** [domain-demand-forecasting.streamlit.app](https://domain-demand-forecasting.streamlit.app/)

Demand forecasting system using ensemble learning across 3 domains:

- **Airline Bookings** — predict flight demand by fare class
- **E-Commerce Demand** — forecast product sales by category
- **Payment Volume** — predict hourly transaction volume

Each domain uses XGBoost + LightGBM as base models with a Ridge regression meta-learner (stacking ensemble) and quantile-based uncertainty bounds.

## Quick Start

```bash
# Clone
git clone https://github.com/sanjitmathur/multi-domain-demand-forecasting.git
cd multi-domain-demand-forecasting

# Setup
python -m venv venv
venv\Scripts\pip install -r requirements.txt   # Windows
# source venv/bin/activate && pip install -r requirements.txt  # Mac/Linux

# Generate data
venv\Scripts\python -m src.data.generate_airline
venv\Scripts\python -m src.data.generate_ecommerce
venv\Scripts\python -m src.data.generate_payment

# Train models
venv\Scripts\python -m src.models.train_all

# Run API (http://localhost:8000)
venv\Scripts\uvicorn api.main:app --port 8000

# Run Dashboard (http://localhost:8501) — open a second terminal
venv\Scripts\streamlit run dashboard/app.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/forecast/airline` | POST | Airline booking forecast |
| `/forecast/ecommerce` | POST | E-commerce demand forecast |
| `/forecast/payment` | POST | Payment volume forecast |
| `/forecast/batch` | POST | Batch forecast across domains |
| `/docs` | GET | Interactive Swagger docs |

**Example request:**

```bash
curl -X POST http://localhost:8000/forecast/airline \
  -H "Content-Type: application/json" \
  -d '{"days_until_departure": 30, "fare_class": "Economy", "competitor_price": 300}'
```

**Response:**

```json
{
  "forecast": 91.18,
  "lower_bound": 84.41,
  "upper_bound": 92.33,
  "confidence": 0.9131
}
```

## Project Structure

```
src/
  data/           # Data generators (airline, ecommerce, payment)
  features/       # Feature engineering pipelines per domain
  models/         # DomainModel, EnsembleForecaster, baselines
  evaluation/     # Metrics (RMSE, MAE, MAPE, SMAPE, R²)
api/              # FastAPI app with prediction endpoints
dashboard/        # Streamlit multi-page app
  pages/          # Model Architecture, Evaluation, Feature Importance, Uncertainty, System Design, Diagnostics, Ablation
data/             # Generated CSV datasets
models/saved/     # Trained model artifacts
results/          # Evaluation results
```

## How It Works

```
Raw Data → Feature Engineering → [XGBoost, LightGBM] → Ridge Meta-Learner → Prediction ± Uncertainty
```

1. **Data Generation** — Synthetic datasets with realistic patterns (seasonality, trends, promotions)
2. **Feature Engineering** — Rolling stats, lag features, cyclical encoding, domain-specific interactions
3. **Base Models** — XGBoost and LightGBM trained independently per domain
4. **Meta-Learner** — Ridge regression on out-of-fold base model predictions (stacking)
5. **Uncertainty** — Quantile regression (10th/90th percentiles) for confidence intervals

## Tech Stack

Python, XGBoost, LightGBM, scikit-learn, SHAP, pandas, FastAPI, Streamlit, Plotly
