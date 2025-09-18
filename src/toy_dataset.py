# scripts/toy_dataset.py
#
# Purpose:
# Create a tiny "toy" dataset so the pipeline can run without
# downloading the full Arrow dataset. Good for demos & CI.

import pandas as pd
from pathlib import Path

# Define a few fake rows with the same columns as the real dataset
rows = [
    {
        "unoptimized": "def add(a, b):\n    return a + b",
        "function_name": "add",
        "unoptimized_size": 20,
    },
    {
        "unoptimized": "def multiply(a, b):\n    return a * b",
        "function_name": "multiply",
        "unoptimized_size": 30,
    },
    {
        "unoptimized": "def divide(a, b):\n    return a / b",
        "function_name": "divide",
        "unoptimized_size": 40,
    },
]

df = pd.DataFrame(rows)

# Save to data/sample/
out_dir = Path(__file__).resolve().parents[1] / "data" / "sample"
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / "toy_dataset.csv"
df.to_csv(out_csv, index=False)

print(f"✅ Sample dataset saved to {out_csv}")
