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

## Running the script

From the repository root, run:

```bash
python src/clean_sales_data.py data/raw/Online\ Retail.xlsx --output-dir data/processed
```

The script writes three outputs:
- cleaned_sales_data.csv for rows retained for sales analysis
- cancelled_orders.csv for cancelled transactions analyzed separately as returns
- rejected_rows.csv for rows removed from the main analysis because they are invalid

A text summary is also written alongside those files.
