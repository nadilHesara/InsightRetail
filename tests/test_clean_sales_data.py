from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clean_sales_data import clean_sales_data


def test_clean_sales_data(tmp_path):
    input_path = tmp_path / "sales.csv"
    df = pd.DataFrame(
        {
            "InvoiceNo": ["536365", "536365", "C536366", "536367", "536368", "536369"],
            "StockCode": ["A1", "A1", "B2", "C3", "D4", "E5"],
            "Description": ["  red  chair ", "red chair", " blue table ", "  ", "green lamp", "yellow bulb"],
            "Quantity": [2, 2, -1, 0, 5, 3],
            "InvoiceDate": ["2020-01-01 10:00:00", "2020-01-01 10:00:00", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            "UnitPrice": [10.0, 10.0, 12.0, 5.0, -1.0, 4.0],
            "CustomerID": [1, None, 2, 3, None, 4],
            "Country": ["UK", "UK", "UK", "UK", "UK", "UK"],
        }
    )
    df.to_csv(input_path, index=False)

    output_dir = tmp_path / "out"
    outputs = clean_sales_data(input_path, output_dir)

    cleaned = outputs["cleaned"]
    cancelled = outputs["cancelled_orders"]
    rejected = outputs["rejected_rows"]

    assert len(cleaned) == 2
    assert len(cancelled) == 1
    assert len(rejected) == 3
    assert pd.api.types.is_datetime64_any_dtype(cleaned["InvoiceDate"])
    assert cleaned["CustomerID"].eq("Unknown").sum() == 1
    assert cleaned["Description"].str.contains("Red Chair").sum() == 1
    assert cleaned["Description"].str.contains("Yellow Bulb").sum() == 1
