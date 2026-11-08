
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT / "data/processed/consolidated_features/"
    "consolidated_molecular_features.csv"
)

FINGERPRINT_FILE = (
    ROOT / "data/processed/molecular_features/"
    "morgan_fingerprints.npz"
)

OUTPUT_DIR = ROOT / "data/processed/stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(value):
    return str(value).strip().lower()


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("Loading molecular features...")

features = pd.read_csv(FEATURE_FILE)

print("Loading Morgan fingerprints...")

data = np.load(FINGERPRINT_FILE, allow_pickle=True)

print("NPZ arrays:", data.files)

fingerprints = data["fingerprints"]
drug_names = data["drug_names"]

# --------------------------------------------------
# 2. BASIC VALIDATION
# --------------------------------------------------

print("\n========== FINGERPRINT SUMMARY ==========")

print("Fingerprint shape:", fingerprints.shape)
print("Fingerprint dtype:", fingerprints.dtype)
print("Number of drug names:", len(drug_names))

if fingerprints.ndim != 2:
    raise ValueError("Fingerprint matrix must be two-dimensional.")

if fingerprints.shape[0] != len(drug_names):
    raise ValueError("Fingerprint count and drug-name count differ.")

if fingerprints.shape[1] != 2048:
    raise ValueError("Expected 2048-bit Morgan fingerprints.")

if not np.isin(fingerprints, [0, 1]).all():
    raise ValueError("Fingerprint matrix contains values other than 0 and 1.")

# --------------------------------------------------
# 3. NORMALIZE DRUG NAMES
# --------------------------------------------------

fp_names = pd.Series(
    [normalize(name) for name in drug_names],
    name="fingerprint_name"
)

feature_names = features["drug_key"].apply(normalize)

print("\nUnique fingerprint names:", fp_names.nunique())
print("Unique feature drug keys:", feature_names.nunique())

# Check duplicates
duplicate_names = fp_names[fp_names.duplicated(keep=False)]

print("Duplicate fingerprint names:", duplicate_names.nunique())

if not duplicate_names.empty:
    print(duplicate_names.drop_duplicates().tolist()[:20])

# --------------------------------------------------
# 4. CHECK NAME MATCHING
# --------------------------------------------------

feature_key_set = set(feature_names)
fingerprint_name_set = set(fp_names)

matched_names = fingerprint_name_set & feature_key_set

missing_from_features = fingerprint_name_set - feature_key_set
missing_from_fingerprints = feature_key_set - fingerprint_name_set

print("\n========== NAME MATCHING ==========")

print("Matched unique names:", len(matched_names))
print("Fingerprint names missing from features:",
      len(missing_from_features))
print("Feature keys missing from fingerprints:",
      len(missing_from_fingerprints))

# --------------------------------------------------
# 5. VERIFY fingerprint_row
# --------------------------------------------------

print("\n========== FINGERPRINT ROW VALIDATION ==========")

row_errors = []

for index, row in features.iterrows():

    if pd.isna(row["fingerprint_row"]):
        if str(row["has_fingerprint"]).lower() in ["true", "1"]:
            row_errors.append(
                (row["drug_name"], "Missing fingerprint_row")
            )
        continue

    fp_index = int(row["fingerprint_row"])

    if fp_index < 0 or fp_index >= len(drug_names):
        row_errors.append(
            (row["drug_name"], "Fingerprint index out of range")
        )
        continue

    expected_name = normalize(row["drug_key"])
    actual_name = normalize(drug_names[fp_index])

    if expected_name != actual_name:
        row_errors.append(
            (
                row["drug_name"],
                f"Expected {expected_name}, found {actual_name}"
            )
        )

print("Fingerprint row mapping errors:", len(row_errors))

for error in row_errors[:20]:
    print(error)

# --------------------------------------------------
# 6. VERIFY FINGERPRINT UNIQUENESS
# --------------------------------------------------

print("\n========== FINGERPRINT CONTENT ==========")

print(
    "Fingerprint rows with all-zero bits:",
    int(np.all(fingerprints == 0, axis=1).sum())
)

print(
    "Unique fingerprint vectors:",
    len(np.unique(fingerprints, axis=0))
)

# --------------------------------------------------
# 7. SAVE VALIDATION REPORT
# --------------------------------------------------

report = {
    "fingerprint_rows": fingerprints.shape[0],
    "fingerprint_dimensions": fingerprints.shape[1],
    "unique_fingerprint_names": fp_names.nunique(),
    "unique_feature_keys": feature_names.nunique(),
    "matched_names": len(matched_names),
    "missing_from_features": len(missing_from_features),
    "missing_from_fingerprints": len(missing_from_fingerprints),
    "duplicate_fingerprint_names": duplicate_names.nunique(),
    "fingerprint_row_errors": len(row_errors),
    "all_zero_fingerprints": int(
        np.all(fingerprints == 0, axis=1).sum()
    ),
}

pd.Series(report).to_csv(
    OUTPUT_DIR / "fingerprint_mapping_validation.csv",
    header=["value"]
)

pd.DataFrame(
    row_errors,
    columns=["drug_name", "error"]
).to_csv(
    OUTPUT_DIR / "fingerprint_mapping_errors.csv",
    index=False
)

print("\n========== FINAL REPORT ==========")

for key, value in report.items():
    print(f"{key}: {value}")

print("\nValidation report saved in:", OUTPUT_DIR)
print("\nFingerprint mapping inspection completed.")
