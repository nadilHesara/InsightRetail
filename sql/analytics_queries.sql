-- Monthly revenue
SELECT DATE_TRUNC('month', invoice_date) AS month, SUM(total_amount) AS revenue
FROM orders
GROUP BY DATE_TRUNC('month', invoice_date)
ORDER BY month;

-- Top 10 products by revenue
SELECT p.product_id, p.description, SUM(oi.sales_amount) AS revenue
FROM order_items AS oi
JOIN products AS p ON oi.product_id = p.product_id
GROUP BY p.product_id, p.description
ORDER BY revenue DESC
LIMIT 10;

-- Top 10 customers by revenue
SELECT c.customer_id, SUM(o.total_amount) AS revenue
FROM orders AS o
JOIN customers AS c ON o.customer_id = c.customer_id
GROUP BY c.customer_id
ORDER BY revenue DESC
LIMIT 10;

-- Revenue by country
SELECT country, SUM(total_amount) AS revenue
FROM orders
GROUP BY country
ORDER BY revenue DESC;

-- Average order value
SELECT AVG(total_amount) AS average_order_value
FROM orders;

-- Revenue by customer segment
SELECT cs.segment, SUM(cs.monetary) AS revenue
FROM customer_segments AS cs
GROUP BY cs.segment
ORDER BY revenue DESC;
