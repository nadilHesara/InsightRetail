CREATE TABLE IF NOT EXISTS customers (
    customer_id BIGINT PRIMARY KEY,
    country VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(50) PRIMARY KEY,
    description TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id BIGINT REFERENCES customers(customer_id),
    invoice_date TIMESTAMP,
    country VARCHAR(100),
    total_amount NUMERIC(12, 2)
);

CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) REFERENCES orders(order_id),
    product_id VARCHAR(50) REFERENCES products(product_id),
    quantity INTEGER,
    unit_price NUMERIC(12, 2),
    sales_amount NUMERIC(12, 2)
);

CREATE TABLE IF NOT EXISTS customer_segments (
    customer_id BIGINT PRIMARY KEY REFERENCES customers(customer_id),
    segment VARCHAR(100),
    recency INTEGER,
    frequency INTEGER,
    monetary NUMERIC(12, 2),
    r_score INTEGER,
    f_score INTEGER,
    m_score INTEGER
);

CREATE TABLE IF NOT EXISTS daily_sales (
    sale_date DATE PRIMARY KEY,
    revenue NUMERIC(12, 2),
    orders_count INTEGER
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id SERIAL PRIMARY KEY,
    forecast_date DATE,
    forecast_value NUMERIC(12, 2),
    model_name VARCHAR(100)
);
