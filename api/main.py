from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.customers import router as customers_router
from api.routes.forecast import router as forecast_router
from api.routes.health import router as health_router
from api.routes.predict import router as predict_router
from api.routes.products import router as products_router
from api.routes.sales import router as sales_router
from api.routes.segments import router as segments_router
from api.routes.summary import router as summary_router


app = FastAPI(
    title="InsightRetail API",
    description="A simple retail analytics backend for InsightRetail with sales, product, customer, and forecast endpoints.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

app.include_router(health_router)
app.include_router(summary_router)
app.include_router(sales_router)
app.include_router(products_router)
app.include_router(customers_router)
app.include_router(segments_router)
app.include_router(forecast_router)
app.include_router(predict_router)
