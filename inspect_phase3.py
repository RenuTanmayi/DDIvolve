
import pandas as pd
from pathlib import Path

files = [
    Path("data/processed/clean_interactions.csv"),
    Path("data/processed/clean_drugs.csv")
]

# Find integrated DDI dataset automatically
integrated_files = list(
    Path("data/processed").glob("integrated_ddi_dataset*.csv")
)

files.extend(integrated_files)

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file)

    if not file.exists():
        print("File not found!")
        continue

    try:
        df = pd.read_csv(file)

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

    except Exception as e:
        print("Error:", e)
