from pathlib import Path

import pandas as pd

from src.rfm_segmentation import build_customer_segments, load_or_build_cleaned_dataset


def test_build_customer_segments(tmp_path):
    raw_path = tmp_path / "retail.xlsx"
    cleaned_path = tmp_path / "cleaned.csv"
    output_path = tmp_path / "segments.csv"

    df = pd.DataFrame(
        {
            "InvoiceNo": ["536365", "536365", "536366", "536367", "C536368"],
            "StockCode": ["A1", "A1", "B2", "C3", "D4"],
            "Description": ["Chair", "Chair", "Table", "Lamp", "Desk"],
            "Quantity": [2, 1, 3, 2, 1],
            "InvoiceDate": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-02-01", "2024-02-10"],
            "UnitPrice": [10, 10, 20, 15, 12],
            "CustomerID": [1, 1, 2, 3, 4],
            "Country": ["UK", "UK", "UK", "UK", "UK"],
        }
    )
    df.to_excel(raw_path, index=False)

    cleaned_df = load_or_build_cleaned_dataset(raw_path=raw_path, cleaned_path=cleaned_path)
    segments = build_customer_segments(cleaned_df, cleaned_path=cleaned_path, output_path=output_path)

    assert len(cleaned_df) == 4
    assert "salesamount" in cleaned_df.columns
    assert {"r_score", "f_score", "m_score", "segment"}.issubset(segments.columns)
    assert output_path.exists()
