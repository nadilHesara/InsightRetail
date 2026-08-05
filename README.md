# InsightRetail data cleaning

This project includes a cleaning pipeline for retail sales data in [src/clean_sales_data.py](src/clean_sales_data.py).

## Cleaning decisions

The following decisions are applied explicitly rather than silently deleting data:

- Duplicate rows were removed from the main analysis dataset.
- Cancelled orders were removed from the sales forecast but preserved separately as product returns in the cancelled orders output.
- Invalid quantities, invalid prices, and invalid or missing invoice dates were moved to a rejected rows file for review.
- Missing customer IDs were filled with the placeholder value "Unknown" instead of being dropped.
- Product descriptions were standardized to a consistent title case format.
- Invoice dates were converted to datetime values where possible; invalid date values were treated as invalid records.

## Running the cleaning script

From the repository root, run:

```bash
python src/clean_sales_data.py data/raw/Online\ Retail.xlsx --output-dir data/processed
```

The script writes three outputs:
- cleaned_sales_data.csv for rows retained for sales analysis
- cancelled_orders.csv for cancelled transactions analyzed separately as returns
- rejected_rows.csv for rows removed from the main analysis because they are invalid

A text summary is also written alongside those files.

## PostgreSQL database setup

The project can load the processed data into PostgreSQL with a simple normalized schema.

### Required processed files
- `data/processed/cleaned_retail_data.csv`
- `data/processed/customer_segments.csv`
- `data/processed/future_30_day_forecast.csv` (optional, used to populate `forecasts`)

### Schema and relationships
- `customers`: one row per customer, keyed by `customer_id`
- `products`: one row per product, keyed by `product_id`
- `orders`: one row per invoice/order, linked to `customers`
- `order_items`: line items linked to `orders` and `products`
- `customer_segments`: segment data linked one-to-one with `customers`
- `daily_sales`: one row per calendar date with revenue and order count
- `forecasts`: future revenue forecasts linked by date and model name

### Setup steps
1. Create a PostgreSQL database, for example `insightretail`.
2. Copy `.env.example` to `.env` and set `DATABASE_URL`.
3. Install the database loader dependencies:

```bash
pip install sqlalchemy psycopg2-binary python-dotenv pandas
```

4. Create the database tables and load the processed data:

```bash
python src/load_database.py
```

If the `forecasts` file is not available, the loader will still populate the other tables and insert a placeholder forecast row.

### Query examples
See `sql/analytics_queries.sql` for example analytics queries.
