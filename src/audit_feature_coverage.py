
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data/processed/consolidated_features/"
    "consolidated_molecular_features.csv"
)

OUTPUT = ROOT / "data/processed/stage2"

df = pd.read_csv(INPUT)

feature_cols = [
    "has_descriptors",
    "has_fingerprint",
    "has_graph",
    "has_2d_structure",
    "has_3d_conformer",
    "has_optimized_3d"
]

print("\n========== FEATURE COVERAGE AUDIT ==========")

for col in feature_cols:
    count = df[col].fillna(False).astype(bool).sum()
    print(f"{col}: {count}/{len(df)}")

# Check missing SMILES
missing_smiles = df[df["smiles"].isna()]

print("\nMissing SMILES:", len(missing_smiles))

# Check drugs with all available feature flags
all_available = df[
    df[feature_cols].fillna(False).all(axis=1)
]

print("All feature flags available:", len(all_available))

# Check individual feature combinations
print("\nFeature availability combinations:")

print(
    df.groupby(feature_cols, dropna=False)
      .size()
      .sort_values(ascending=False)
      .to_string()
)

# Save detailed audit
df[["drug_name", "drug_key"] + feature_cols].to_csv(
    OUTPUT / "feature_coverage_audit.csv",
    index=False
)

# Save records missing any feature
df[
    ~df[feature_cols].fillna(False).all(axis=1)
].to_csv(
    OUTPUT / "incomplete_feature_records.csv",
    index=False
)

print("\nSaved:")
print("feature_coverage_audit.csv")
print("incomplete_feature_records.csv")

print("\n========== AUDIT COMPLETE ==========")
