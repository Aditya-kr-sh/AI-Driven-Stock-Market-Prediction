# FastAPI Backend Service

FastAPI-powered REST API serving prediction endpoints, portfolio optimization target weights, active model status checks, and health monitors.

## API Endpoints
- `POST /predict`: Generates stock price predictions utilizing XGBoost, LSTM, and Transformer model instances.
- `POST /portfolio`: Suggests capital allocations using Monte Carlo simulations.
- `GET /models`: Returns active model metrics and configurations.
- `GET /health`: Basic operational validation check.
