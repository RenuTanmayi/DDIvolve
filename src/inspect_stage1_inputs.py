
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

files_to_check = [
    ROOT / "data/processed/consolidated_features/consolidated_molecular_features.csv",
    ROOT / "data/processed/integrated_ddi_dataset.csv",
    ROOT / "data/processed/interaction_template_catalog.csv",
    ROOT / "data/interactions.csv",
]

# Also discover other possible interaction datasets.
for folder in [ROOT / "data/processed", ROOT / "data"]:
    if folder.exists():
        for path in folder.glob("*.csv"):
            name = path.name.lower()
            if any(word in name for word in ["interaction", "ddi"]):
                files_to_check.append(path)

files_to_check = list(dict.fromkeys(files_to_check))

for path in files_to_check:
    print("\n" + "=" * 75)
    print("FILE:", path.relative_to(ROOT))

    if not path.exists():
        print("STATUS: NOT FOUND")
        continue

    try:
        df = pd.read_csv(path, nrows=5)
        total_rows = sum(1 for _ in open(path, encoding="utf-8-sig")) - 1

        print("STATUS: FOUND")
        print("TOTAL ROWS:", total_rows)
        print("COLUMNS:", len(df.columns))
        print("\nCOLUMN NAMES:")
        for col in df.columns:
            print(" -", col)

        print("\nSAMPLE RECORDS:")
        print(df.head(3).to_string(index=False))

        print("\nMISSING VALUES IN SAMPLE:")
        print(df.isnull().sum().to_string())

    except Exception as e:
        print("ERROR:", e)

print("\nInspection completed.")
