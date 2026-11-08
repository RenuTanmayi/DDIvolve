
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT / "data/processed/consolidated_features/"
    "consolidated_molecular_features.csv"
)

INTERACTION_FILE = ROOT / "data/processed/clean_interactions.csv"

OUTPUT_DIR = ROOT / "data/processed/stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Molecular descriptors to use initially
DESCRIPTOR_COLUMNS = [
    "molecular_weight",
    "exact_molecular_weight",
    "logp",
    "h_bond_donors",
    "h_bond_acceptors",
    "tpsa",
    "rotatable_bonds",
    "qed",
    "atom_count",
    "bond_count",
    "ring_count",
    "aromatic_rings",
    "aliphatic_rings",
    "fraction_csp3",
    "heavy_atom_count",
    "formal_charge",
]

# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

print("\nLoading datasets...")

features = pd.read_csv(FEATURE_FILE)
interactions = pd.read_csv(INTERACTION_FILE)

print("Molecular feature records:", len(features))
print("Interaction records:", len(interactions))

# --------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_feature_columns = [
    "drug_key",
    "drug_name",
    "has_descriptors",
    "has_fingerprint",
    "fingerprint_row",
] + DESCRIPTOR_COLUMNS

required_interaction_columns = [
    "Drug 1",
    "Drug 2",
    "Interaction Description",
    "drug1_key",
    "drug2_key",
    "pair_id",
]

for col in required_feature_columns:
    if col not in features.columns:
        raise ValueError(f"Missing feature column: {col}")

for col in required_interaction_columns:
    if col not in interactions.columns:
        raise ValueError(f"Missing interaction column: {col}")

# --------------------------------------------------
# 4. NORMALIZE DRUG KEYS
# --------------------------------------------------

def normalize_key(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

features["drug_key"] = normalize_key(features["drug_key"])
interactions["drug1_key"] = normalize_key(interactions["drug1_key"])
interactions["drug2_key"] = normalize_key(interactions["drug2_key"])

# Remove invalid keys
features = features.dropna(subset=["drug_key"]).copy()

if features["drug_key"].duplicated().any():
    duplicates = features[
        features["drug_key"].duplicated(keep=False)
    ]
    duplicates.to_csv(
        OUTPUT_DIR / "duplicate_feature_keys.csv",
        index=False
    )
    raise ValueError(
        "Duplicate molecular drug keys found. "
        "Resolve them before continuing."
    )

# --------------------------------------------------
# 5. PREPARE MOLECULAR FEATURE LOOKUP
# --------------------------------------------------

lookup_columns = [
    "drug_key",
    "drug_name",
    "smiles",
    "canonical_smiles",
    "has_descriptors",
    "has_fingerprint",
    "fingerprint_row",
] + DESCRIPTOR_COLUMNS

feature_lookup = features[lookup_columns].copy()

# Prefix molecular columns for each drug
drug1_lookup = feature_lookup.rename(
    columns={
        col: f"drug1_{col}"
        for col in feature_lookup.columns
        if col != "drug_key"
    }
).rename(columns={"drug_key": "drug1_key"})

drug2_lookup = feature_lookup.rename(
    columns={
        col: f"drug2_{col}"
        for col in feature_lookup.columns
        if col != "drug_key"
    }
).rename(columns={"drug_key": "drug2_key"})

# --------------------------------------------------
# 6. MATCH DRUG 1 AND DRUG 2
# --------------------------------------------------

print("\nMatching Drug 1...")

merged = interactions.merge(
    drug1_lookup,
    on="drug1_key",
    how="left",
    validate="many_to_one"
)

print("Matching Drug 2...")

merged = merged.merge(
    drug2_lookup,
    on="drug2_key",
    how="left",
    validate="many_to_one"
)

# --------------------------------------------------
# 7. CHECK MATCHING STATUS
# --------------------------------------------------

merged["drug1_matched"] = merged["drug1_drug_name"].notna()
merged["drug2_matched"] = merged["drug2_drug_name"].notna()

merged["both_drugs_matched"] = (
    merged["drug1_matched"] &
    merged["drug2_matched"]
)

print("\n========== MATCHING SUMMARY ==========")

print("Total interactions:", len(merged))
print("Drug 1 matched:", merged["drug1_matched"].sum())
print("Drug 2 matched:", merged["drug2_matched"].sum())
print("Both matched:", merged["both_drugs_matched"].sum())
print("Unmatched interactions:", (~merged["both_drugs_matched"]).sum())

# Save unmatched interactions for inspection
unmatched = merged[~merged["both_drugs_matched"]].copy()

unmatched.to_csv(
    OUTPUT_DIR / "unmatched_interactions.csv",
    index=False
)

# --------------------------------------------------
# 8. CHECK FEATURE AVAILABILITY
# --------------------------------------------------

def is_true(series):
    return series.astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

merged["drug1_has_descriptors"] = is_true(
    merged["drug1_has_descriptors"]
)

merged["drug2_has_descriptors"] = is_true(
    merged["drug2_has_descriptors"]
)

merged["drug1_has_fingerprint"] = is_true(
    merged["drug1_has_fingerprint"]
)

merged["drug2_has_fingerprint"] = is_true(
    merged["drug2_has_fingerprint"]
)

merged["drug1_descriptors_complete"] = (
    merged[[f"drug1_{c}" for c in DESCRIPTOR_COLUMNS]]
    .notna().all(axis=1)
)

merged["drug2_descriptors_complete"] = (
    merged[[f"drug2_{c}" for c in DESCRIPTOR_COLUMNS]]
    .notna().all(axis=1)
)

merged["features_available"] = (
    merged["both_drugs_matched"] &
    merged["drug1_has_descriptors"] &
    merged["drug2_has_descriptors"] &
    merged["drug1_has_fingerprint"] &
    merged["drug2_has_fingerprint"] &
    merged["drug1_descriptors_complete"] &
    merged["drug2_descriptors_complete"]
)

# --------------------------------------------------
# 9. CREATE FINAL DATASETS
# --------------------------------------------------

# All interactions where both drugs have molecular records
matched = merged[
    merged["both_drugs_matched"]
].copy()

matched.to_csv(
    OUTPUT_DIR / "stage1_all_matched.csv",
    index=False
)

# Model-ready interactions with required features
model_ready = merged[
    merged["features_available"]
].copy()

model_ready.to_csv(
    OUTPUT_DIR / "stage1_model_ready.csv",
    index=False
)

# --------------------------------------------------
# 10. INTERACTION LABEL INSPECTION
# --------------------------------------------------

print("\n========== INTERACTION DESCRIPTION ANALYSIS ==========")

print(
    "Unique descriptions:",
    interactions["Interaction Description"].nunique()
)

description_counts = (
    interactions["Interaction Description"]
    .value_counts()
)

description_counts.head(50).to_csv(
    OUTPUT_DIR / "interaction_description_distribution.csv",
    header=["record_count"]
)

# --------------------------------------------------
# 11. DUPLICATE PAIR ANALYSIS
# --------------------------------------------------

pair_counts = (
    matched.groupby("pair_id")
    .size()
    .sort_values(ascending=False)
)

print("Unique matched pair IDs:", len(pair_counts))
print("Pairs with multiple records:", (pair_counts > 1).sum())

print(
    "Exact duplicate interaction rows:",
    interactions.duplicated(
        subset=[
            "drug1_key",
            "drug2_key",
            "Interaction Description"
        ]
    ).sum()
)

# --------------------------------------------------
# 12. SUMMARY
# --------------------------------------------------

summary = {
    "total_interactions": len(interactions),
    "total_molecular_records": len(features),
    "both_drugs_matched": int(merged["both_drugs_matched"].sum()),
    "unmatched_interactions": int((~merged["both_drugs_matched"]).sum()),
    "all_matched_records": len(matched),
    "model_ready_records": len(model_ready),
    "unique_interaction_descriptions": int(
        interactions["Interaction Description"].nunique()
    ),
    "unique_matched_pairs": int(matched["pair_id"].nunique()),
    "pairs_with_multiple_records": int((pair_counts > 1).sum()),
}

pd.Series(summary).to_csv(
    OUTPUT_DIR / "stage1_validation_summary.csv",
    header=["value"]
)

print("\n========== FINAL STAGE 1 SUMMARY ==========")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nGenerated files:")
for path in OUTPUT_DIR.iterdir():
    print("-", path.name)

print("\nStage 1 preparation completed.")
