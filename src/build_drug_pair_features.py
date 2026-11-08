
from pathlib import Path
import numpy as np
import pandas as pd

# ==================================================
# 1. PATHS AND CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT / "data/processed/consolidated_features/"
    "consolidated_molecular_features.csv"
)

LABEL_FILE = (
    ROOT / "data/processed/stage1/"
    "template_label_analysis.csv"
)

FINGERPRINT_FILE = (
    ROOT / "data/processed/molecular_features/"
    "morgan_fingerprints.npz"
)

OUTPUT_DIR = ROOT / "data/processed/stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 5000

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

# ==================================================
# 2. LOAD DATASETS
# ==================================================

print("\nLoading datasets...")

features = pd.read_csv(FEATURE_FILE)
labels = pd.read_csv(LABEL_FILE)

fp_data = np.load(FINGERPRINT_FILE, allow_pickle=True)

fingerprints = fp_data["fingerprints"]
fp_names = fp_data["drug_names"]

print("Molecular records:", len(features))
print("Interaction records:", len(labels))
print("Fingerprint matrix:", fingerprints.shape)

# ==================================================
# 3. VALIDATE INPUT SCHEMAS
# ==================================================

required_labels = [
    "drug1_key",
    "drug2_key",
    "Drug 1",
    "Drug 2",
    "Interaction Description",
    "pair_id",
    "interaction_family",
    "direction",
    "affected_drug",
]

required_features = [
    "drug_key",
    "drug_name",
    "fingerprint_row",
    "has_fingerprint",
    "has_descriptors",
] + DESCRIPTOR_COLUMNS

for col in required_labels:
    if col not in labels.columns:
        raise ValueError(f"Missing label column: {col}")

for col in required_features:
    if col not in features.columns:
        raise ValueError(f"Missing feature column: {col}")

if fingerprints.shape != (len(fp_names), 2048):
    raise ValueError("Invalid fingerprint dimensions.")

# ==================================================
# 4. NORMALIZE DRUG KEYS
# ==================================================

def normalize(series):
    return series.astype("string").str.strip().str.lower()

features["drug_key"] = normalize(features["drug_key"])
labels["drug1_key"] = normalize(labels["drug1_key"])
labels["drug2_key"] = normalize(labels["drug2_key"])

fp_names = np.array(
    [str(name).strip().lower() for name in fp_names]
)

# ==================================================
# 5. PREPARE FEATURE LOOKUP
# ==================================================

if features["drug_key"].duplicated().any():
    raise ValueError("Duplicate molecular drug keys found.")

feature_lookup = features[
    ["drug_key", "drug_name", "fingerprint_row",
     "has_fingerprint", "has_descriptors"]
    + DESCRIPTOR_COLUMNS
].copy()

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

# ==================================================
# 6. MATCH BOTH DRUGS
# ==================================================

print("\nMatching molecular features...")

data = labels.merge(
    drug1_lookup,
    on="drug1_key",
    how="left",
    validate="many_to_one"
)

data = data.merge(
    drug2_lookup,
    on="drug2_key",
    how="left",
    validate="many_to_one"
)

# ==================================================
# 7. VALIDATE DESCRIPTORS AND FINGERPRINTS
# ==================================================

def true_mask(series):
    return series.astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

data["valid_drug1"] = (
    data["drug1_drug_name"].notna() &
    true_mask(data["drug1_has_descriptors"]) &
    true_mask(data["drug1_has_fingerprint"])
)

data["valid_drug2"] = (
    data["drug2_drug_name"].notna() &
    true_mask(data["drug2_has_descriptors"]) &
    true_mask(data["drug2_has_fingerprint"])
)

for drug in ["drug1", "drug2"]:
    cols = [f"{drug}_{col}" for col in DESCRIPTOR_COLUMNS]

    data[f"{drug}_complete"] = (
        data[cols].notna().all(axis=1)
    )

    fp_rows = pd.to_numeric(
        data[f"{drug}_fingerprint_row"],
        errors="coerce"
    )

    valid_indices = (
        fp_rows.notna() &
        (fp_rows >= 0) &
        (fp_rows < len(fp_names))
    )

    data[f"{drug}_valid_fp_row"] = valid_indices

    # Verify that the fingerprint row belongs to the drug.
    positions = np.flatnonzero(valid_indices)

    for pos in positions:
        fp_index = int(fp_rows.iloc[pos])
        expected_key = data.iloc[pos][f"{drug}_key"]

        if fp_names[fp_index] != expected_key:
            raise ValueError(
                f"Fingerprint mismatch for {expected_key}"
            )

data["valid_pair"] = (
    data["valid_drug1"] &
    data["valid_drug2"] &
    data["drug1_complete"] &
    data["drug2_complete"] &
    data["drug1_valid_fp_row"] &
    data["drug2_valid_fp_row"]
)

# Interaction labels must be present.
label_columns = [
    "interaction_family",
    "direction",
    "affected_drug",
]

data["valid_labels"] = data[label_columns].notna().all(axis=1)

valid = data[
    data["valid_pair"] & data["valid_labels"]
].copy().reset_index(drop=True)

print("\n========== FEATURE AVAILABILITY ==========")

print("Original interactions:", len(data))
print("Valid molecular pairs:", data["valid_pair"].sum())
print("Valid labels:", data["valid_labels"].sum())
print("Final eligible records:", len(valid))

# ==================================================
# 8. PREPARE NUMPY FEATURE MATRICES
# ==================================================

n = len(valid)

if n == 0:
    raise ValueError("No eligible drug pairs found.")

descriptor_names = (
    [f"drug1_{c}" for c in DESCRIPTOR_COLUMNS] +
    [f"drug2_{c}" for c in DESCRIPTOR_COLUMNS] +
    [f"abs_diff_{c}" for c in DESCRIPTOR_COLUMNS] +
    [f"product_{c}" for c in DESCRIPTOR_COLUMNS]
)

fingerprint_names = (
    [f"drug1_fp_{i}" for i in range(2048)] +
    [f"drug2_fp_{i}" for i in range(2048)] +
    [f"xor_fp_{i}" for i in range(2048)] +
    [f"and_fp_{i}" for i in range(2048)]
)

# Separate files prevent one huge dense combined matrix.
X_desc = np.lib.format.open_memmap(
    OUTPUT_DIR / "pair_descriptors.npy",
    mode="w+",
    dtype=np.float32,
    shape=(n, 64)
)

X_fp = np.lib.format.open_memmap(
    OUTPUT_DIR / "pair_fingerprints.npy",
    mode="w+",
    dtype=np.uint8,
    shape=(n, 8192)
)

# ==================================================
# 9. CONSTRUCT PAIR FEATURES IN CHUNKS
# ==================================================

print("\nConstructing pair features...")

for start in range(0, n, CHUNK_SIZE):

    end = min(start + CHUNK_SIZE, n)

    batch = valid.iloc[start:end]

    # ---------- Molecular descriptors ----------

    A = batch[
        [f"drug1_{c}" for c in DESCRIPTOR_COLUMNS]
    ].to_numpy(dtype=np.float32)

    B = batch[
        [f"drug2_{c}" for c in DESCRIPTOR_COLUMNS]
    ].to_numpy(dtype=np.float32)

    abs_diff = np.abs(A - B)
    product = A * B

    descriptor_matrix = np.concatenate(
        [A, B, abs_diff, product],
        axis=1
    )

    X_desc[start:end] = descriptor_matrix

    # ---------- Morgan fingerprints ----------

    idx1 = batch["drug1_fingerprint_row"].to_numpy(
        dtype=np.int64
    )

    idx2 = batch["drug2_fingerprint_row"].to_numpy(
        dtype=np.int64
    )

    fp_A = fingerprints[idx1]
    fp_B = fingerprints[idx2]

    fp_xor = np.bitwise_xor(fp_A, fp_B)
    fp_and = np.bitwise_and(fp_A, fp_B)

    fingerprint_matrix = np.concatenate(
        [fp_A, fp_B, fp_xor, fp_and],
        axis=1
    )

    X_fp[start:end] = fingerprint_matrix

    if start % (CHUNK_SIZE * 5) == 0:
        print(f"Processed {end}/{n} interactions")

X_desc.flush()
X_fp.flush()

# ==================================================
# 10. CREATE CANONICAL PAIR GROUP IDs
# ==================================================

# Preserve original Drug 1 / Drug 2 ordering in features.
# Use unordered pair IDs only for leakage prevention.

valid["canonical_pair_id"] = valid.apply(
    lambda row: " || ".join(
        sorted([row["drug1_key"], row["drug2_key"]])
    ),
    axis=1
)

# ==================================================
# 11. SAVE LABELS AND METADATA
# ==================================================

output_columns = [
    "Drug 1",
    "Drug 2",
    "drug1_key",
    "drug2_key",
    "pair_id",
    "canonical_pair_id",
    "Interaction Description",
    "interaction_family",
    "direction",
    "affected_drug",
]

valid[output_columns].to_csv(
    OUTPUT_DIR / "pair_labels.csv",
    index=False
)

pd.Series(descriptor_names).to_csv(
    OUTPUT_DIR / "descriptor_feature_names.csv",
    index=False,
    header=["feature_name"]
)

pd.Series(fingerprint_names).to_csv(
    OUTPUT_DIR / "fingerprint_feature_names.csv",
    index=False,
    header=["feature_name"]
)

# ==================================================
# 12. FINAL VALIDATION
# ==================================================

print("\n========== STAGE 2 VALIDATION ==========")

print("Descriptor matrix:", X_desc.shape)
print("Fingerprint matrix:", X_fp.shape)

print("Descriptor dtype:", X_desc.dtype)
print("Fingerprint dtype:", X_fp.dtype)

print("Descriptor missing values:", np.isnan(X_desc).sum())

print("Fingerprint values valid:",
      np.isin(X_fp, [0, 1]).all())

print("Unique interaction families:",
      valid["interaction_family"].nunique())

print("Unique directions:",
      valid["direction"].nunique())

print("Unique affected-drug labels:",
      valid["affected_drug"].nunique())

print("Unique canonical drug pairs:",
      valid["canonical_pair_id"].nunique())

print("\n========== LABEL DISTRIBUTION ==========")

print(valid["interaction_family"].value_counts())

print("\n========== GENERATED FILES ==========")

for file in sorted(OUTPUT_DIR.iterdir()):
    if file.is_file():
        print(f"{file.name}: {file.stat().st_size / (1024**2):.2f} MB")

print("\nStage 2 drug-pair feature construction completed.")
