# InsightRetail

InsightRetail is a university portfolio project for retail sales forecasting and analytics. It combines data cleaning, customer segmentation, model training, a PostgreSQL analytics pipeline, a FastAPI backend, and a Streamlit dashboard.

## Main features

- Retail data cleaning and validation
- Customer segmentation and revenue analysis
- Forecasting with trained regression and time-series models
- PostgreSQL-backed analytics queries
- FastAPI backend with sales, products, customers, segments, forecast, and prediction endpoints
- Streamlit dashboard for interactive visualization

## Dataset source

The project uses retail transaction data stored in `data/raw/` and processed into `data/processed/`. The data format follows an Online Retail-style dataset with invoices, customers, products, and sales amounts.

## Architecture

```mermaid
flowchart LR
    A[User / Analyst] -->|browser| B[Streamlit dashboard]
    A -->|HTTP| C[FastAPI backend]
    C -->|SQL| D[PostgreSQL database]
    C -->|model file| E[Trained forecast model]
    B -->|local data| F[Processed CSV files]
    B -->|optionally| C
    D ---|data ingestion| F
``` 

## Folder structure

- `api/` — FastAPI backend and route definitions
- `dashboard/` — Streamlit dashboard app
- `data/` — raw and processed datasets
- `models/` — trained model artifacts and metrics
- `sql/` — database schema and analytics queries
- `src/` — data cleaning, training, forecasting, and loading scripts
- `tests/` — pytest coverage for project components
- `README.md` — project documentation

## Installation

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Database setup

1. Copy `.env.example` to `.env`.
2. Edit `.env` and set your `DATABASE_URL`.
3. Run the database loader:

```bash
python src/load_database.py
```

### Example `.env`

```ini
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/insightretail
```

## Model training

Train or retrain the forecasting model with:

```bash
python src/train.py
```

## FastAPI backend

Run the API locally with:

```bash
uvicorn api.main:app --reload
```

Then visit:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## Streamlit dashboard

Run the dashboard locally with:

```bash
streamlit run dashboard/app.py
```

## Docker deployment

Build and start all services with:

```bash
docker compose up --build
```

Or on older systems:

```powershell
docker-compose up --build
```

## Evaluation metrics

The project reports model metrics such as MAE, RMSE, and MAPE. Metrics are saved to `models/model_metrics.json` after training.

## Screenshots

Add project screenshots here once the dashboard and results are generated.

## Limitations

- The dashboard depends on local processed CSV files.
- The API assumes PostgreSQL data has already been loaded.
- The forecasting model may require retraining for new datasets.

## Future improvements

- Add authentication for API access
- Enable dynamic model retraining from the dashboard
- Add more production-ready logging and monitoring
- Expand the dataset ingestion pipeline for new retail sources

## FastAPI backend

This project includes a simple backend API under `api/` for summary, sales, product, customer, segment, forecast, and prediction endpoints.

### Run the API

From the repository root:

```bash
uvicorn api.main:app --reload
```

### Available endpoints

- `GET /health`
- `GET /summary`
- `GET /sales/daily`
- `GET /products/top`
- `GET /customers/top`
- `GET /segments`
- `GET /forecast`
- `POST /predict`

### Environment configuration

Create a `.env` file from `.env.example` and set the PostgreSQL `DATABASE_URL`.

Example:

```ini
DATABASE_URL=postgresql://user:password@localhost:5432/insightretail
```

### Tests

Run the FastAPI tests with:

```bash
pytest tests/test_api.py -q
```

## Streamlit dashboard

Run the dashboard locally with:

```bash
streamlit run dashboard/app.py
```

### Streamlit deployment on Streamlit Cloud

1. Ensure your repository contains `dashboard/app.py`, `.streamlit/config.toml`, and `requirements.txt`.
2. Deploy the repository to Streamlit Cloud.
3. In Streamlit Cloud app settings, set environment variables as needed.

If you need a public dashboard demo, Streamlit Cloud is a good fit for this project because it can host the dashboard directly from the repo.
