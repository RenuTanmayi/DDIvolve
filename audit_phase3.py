
import pandas as pd
from pathlib import Path

base = Path("data/processed")

drugs = pd.read_csv(base / "clean_drugs.csv")
interactions = pd.read_csv(base / "clean_interactions.csv")
df = pd.read_csv(base / "integrated_ddi_dataset.csv")

print("\n========== DRUG DATASET ==========")

print("Total drug records:", len(drugs))
print("Successful lookups:",
      (drugs["lookup_status"] == "success").sum())

print("Valid SMILES:",
      drugs["smiles_validated"].fillna(False).astype(bool).sum())

print("Missing SMILES:",
      drugs["canonical_smiles"].isna().sum())

print("\n========== INTEGRATED DATASET ==========")

print("Total interaction records:", len(df))

print("Drug 1 molecular records:",
      df["drug1_canonical_smiles"].notna().sum())

print("Drug 2 molecular records:",
      df["drug2_canonical_smiles"].notna().sum())

both = (
    df["drug1_canonical_smiles"].notna() &
    df["drug2_canonical_smiles"].notna()
)

print("Both drugs have molecular structures:", both.sum())

print("Incomplete molecular records:", (~both).sum())

print("\n========== INTERACTION DESCRIPTIONS ==========")

print("Unique descriptions:",
      interactions["Interaction Description"].nunique())

print("\nMost frequent descriptions:")

print(
    interactions["Interaction Description"]
    .value_counts()
    .head(15)
    .to_string()
)

print("\n========== DUPLICATE ANALYSIS ==========")

print("Duplicate pair IDs:",
      df["pair_id"].duplicated().sum())

print("Unique drug pairs:",
      df["pair_id"].nunique())

print("\n========== INTERACTION CATEGORY EXPLORATION ==========")

print(
    interactions["Interaction Description"]
    .str.extract(r"may (.*?) the")[0]
    .value_counts()
    .head(20)
    .to_string()
)

print("\nDataset audit completed.")
