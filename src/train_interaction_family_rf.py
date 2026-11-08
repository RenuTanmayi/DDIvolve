
from pathlib import Path
import time
import json
import gc
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack, vstack

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "stage2"

LABEL_FILE = DATA_DIR / "pair_labels_with_affected_drug_key.csv"
VALIDATED_FILE = DATA_DIR / "validated_interaction_labels.csv"

DESC_FILE = DATA_DIR / "pair_descriptors.npy"
FP_FILE = DATA_DIR / "pair_fingerprints.npy"

DESC_NAMES_FILE = DATA_DIR / "descriptor_feature_names.csv"
FP_NAMES_FILE = DATA_DIR / "fingerprint_feature_names.csv"

OUTPUT_DIR = ROOT / "models" / "stage2_random_forest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Full dataset only
RANDOM_STATE = 42

# Seven folds:
# 5 folds training, 1 validation, 1 testing
N_SPLITS = 7

# Random Forest configuration
N_ESTIMATORS = 30
MAX_DEPTH = 18
MIN_SAMPLES_LEAF = 3
MAX_FEATURES = "sqrt"

# Fingerprint conversion chunk size
CHUNK_SIZE = 2000

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n========== LOADING DATA ==========")

start_time = time.time()

required_files = [
    LABEL_FILE,
    VALIDATED_FILE,
    DESC_FILE,
    FP_FILE,
    DESC_NAMES_FILE,
    FP_NAMES_FILE
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

labels = pd.read_csv(LABEL_FILE)
validated = pd.read_csv(VALIDATED_FILE)

descriptors = np.load(
    DESC_FILE,
    mmap_mode="r"
)

fingerprints = np.load(
    FP_FILE,
    mmap_mode="r"
)

print("Labels:", labels.shape)
print("Validated labels:", validated.shape)
print("Descriptors:", descriptors.shape)
print("Fingerprints:", fingerprints.shape)

# ============================================================
# 3. DATA AND FEATURE ALIGNMENT VALIDATION
# ============================================================

print("\n========== ALIGNMENT VALIDATION ==========")

alignment_columns = [
    "drug1_key",
    "drug2_key",
    "canonical_pair_id",
    "interaction_family",
    "direction",
    "affected_drug_key"
]

for col in alignment_columns:
    if col not in labels.columns:
        raise ValueError(
            f"Missing column in labels: {col}"
        )

    if col not in validated.columns:
        raise ValueError(
            f"Missing column in validated labels: {col}"
        )

if len(labels) != len(validated):
    raise ValueError(
        "Labels and validated labels have different row counts."
    )

if not labels[alignment_columns].fillna("").equals(
    validated[alignment_columns].fillna("")
):
    raise ValueError(
        "Label alignment verification failed."
    )

if len(labels) != descriptors.shape[0]:
    raise ValueError(
        "Descriptor rows do not match label rows."
    )

if len(labels) != fingerprints.shape[0]:
    raise ValueError(
        "Fingerprint rows do not match label rows."
    )

if descriptors.ndim != 2 or descriptors.shape[1] != 64:
    raise ValueError(
        f"Expected 64 descriptors, got {descriptors.shape}"
    )

if fingerprints.ndim != 2 or fingerprints.shape[1] != 8192:
    raise ValueError(
        f"Expected 8192 fingerprint features, got {fingerprints.shape}"
    )

if not np.isfinite(descriptors).all():
    raise ValueError(
        "Descriptors contain NaN or infinite values."
    )

if labels["interaction_family"].isna().any():
    raise ValueError(
        "Missing interaction family labels detected."
    )

if labels["canonical_pair_id"].isna().any():
    raise ValueError(
        "Missing canonical pair IDs detected."
    )

print("All alignment checks passed.")

# ============================================================
# 4. PREPARE TARGET AND GROUPS
# ============================================================

print("\n========== PREPARING TARGET ==========")

y = labels["interaction_family"].astype(str).to_numpy()

groups = labels["canonical_pair_id"].astype(str).to_numpy()

classes = np.sort(np.unique(y))

print("Total interaction records:", len(y))
print("Unique canonical pairs:", len(np.unique(groups)))
print("Interaction families:", len(classes))

print("\nOriginal class distribution:")

class_distribution = (
    labels.groupby("interaction_family")
    .agg(
        records=("interaction_family", "size"),
        unique_pairs=("canonical_pair_id", "nunique")
    )
    .sort_values("records", ascending=False)
)

print(class_distribution.to_string())

# ============================================================
# 5. VALIDATE GROUP COUNTS
# ============================================================

print("\n========== GROUP VALIDATION ==========")

group_counts = (
    labels.groupby("interaction_family")[
        "canonical_pair_id"
    ].nunique()
)

min_groups = int(group_counts.min())

print("Minimum unique groups in a class:", min_groups)

if min_groups < 3:
    raise ValueError(
        "At least one class has fewer than 3 unique drug pairs. "
        "Reliable grouped train/validation/test splitting "
        "is not possible."
    )

# Use no more folds than the rarest class's unique groups.
n_splits = min(N_SPLITS, min_groups)

if n_splits < N_SPLITS:
    warnings.warn(
        f"Reducing folds from {N_SPLITS} to {n_splits} "
        "because of rare interaction families."
    )

# ============================================================
# 6. GROUP-AWARE STRATIFIED SPLITTING
# ============================================================

print("\n========== DATASET SPLITTING ==========")

splitter = StratifiedGroupKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_STATE
)

folds = list(
    splitter.split(
        np.zeros(len(y)),
        y,
        groups
    )
)

# With 7 folds:
# Fold 0 -> Test
# Fold 1 -> Validation
# Folds 2-6 -> Training

test_idx = folds[0][1]
val_idx = folds[1][1]

train_idx = np.concatenate(
    [folds[i][1] for i in range(2, n_splits)]
)

# Sort indices for reproducibility
train_idx = np.sort(train_idx)
val_idx = np.sort(val_idx)
test_idx = np.sort(test_idx)

# Verify complete and non-overlapping record allocation
all_indices = np.concatenate([
    train_idx,
    val_idx,
    test_idx
])

if len(all_indices) != len(labels):
    raise AssertionError(
        "Some dataset records are missing from the splits."
    )

if len(np.unique(all_indices)) != len(labels):
    raise AssertionError(
        "Duplicate record indices found across splits."
    )

# Verify no canonical pair leakage
train_groups = set(groups[train_idx])
val_groups = set(groups[val_idx])
test_groups = set(groups[test_idx])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print("Training records:", len(train_idx))
print("Validation records:", len(val_idx))
print("Testing records:", len(test_idx))

print(
    "\nActual proportions:",
    f"Train={len(train_idx)/len(y):.2%},",
    f"Validation={len(val_idx)/len(y):.2%},",
    f"Test={len(test_idx)/len(y):.2%}"
)

print("\nUnique groups per split:")
print("Train:", len(train_groups))
print("Validation:", len(val_groups))
print("Test:", len(test_groups))

print("\nNo canonical pair leakage detected.")

# ============================================================
# 7. SAVE SPLIT INDICES
# ============================================================

np.savez_compressed(
    OUTPUT_DIR / "split_indices.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

# ============================================================
# 8. CLASS DISTRIBUTION AUDIT
# ============================================================

print("\n========== CLASS DISTRIBUTION AUDIT ==========")

split_distribution = []

for split_name, indices in [
    ("Train", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx)
]:

    split_labels = y[indices]
    split_groups = groups[indices]

    for family in classes:

        mask = split_labels == family

        split_distribution.append({
            "split": split_name,
            "interaction_family": family,
            "records": int(mask.sum()),
            "unique_pairs": int(
                len(np.unique(split_groups[mask]))
            )
        })

split_audit = pd.DataFrame(split_distribution)

split_audit.to_csv(
    OUTPUT_DIR / "split_class_distribution.csv",
    index=False
)

print(
    split_audit.pivot(
        index="interaction_family",
        columns="split",
        values="records"
    ).fillna(0).to_string()
)

# Ensure every class is represented in training.
missing_train_classes = (
    set(classes) - set(np.unique(y[train_idx]))
)

if missing_train_classes:
    raise ValueError(
        f"Classes missing from training: {missing_train_classes}"
    )

# Rare classes can be absent from validation or testing.
for split_name, indices in [
    ("Validation", val_idx),
    ("Test", test_idx)
]:

    missing = set(classes) - set(np.unique(y[indices]))

    if missing:
        warnings.warn(
            f"{split_name} does not contain these classes: "
            f"{sorted(missing)}. Their metrics cannot be "
            "reliably evaluated in this split."
        )

# ============================================================
# 9. PREPARE MOLECULAR FEATURES
# ============================================================

print("\n========== PREPARING MOLECULAR FEATURES ==========")

# Molecular descriptors
X_desc = csr_matrix(
    np.asarray(
        descriptors,
        dtype=np.float32
    )
)

print("Descriptor matrix:", X_desc.shape)

# Convert fingerprints in chunks to avoid creating
# the entire dense float32 fingerprint matrix in RAM.

fingerprint_chunks = []

for start in range(0, len(labels), CHUNK_SIZE):

    end = min(
        start + CHUNK_SIZE,
        len(labels)
    )

    chunk = np.asarray(
        fingerprints[start:end],
        dtype=np.uint8
    )

    fingerprint_chunks.append(
        csr_matrix(chunk, dtype=np.float32)
    )

    if start % 20000 == 0:
        print(
            f"Fingerprint conversion: {end}/{len(labels)}"
        )

X_fp = vstack(
    fingerprint_chunks,
    format="csr"
)

del fingerprint_chunks
gc.collect()

print("Fingerprint matrix:", X_fp.shape)

# Combine descriptors and fingerprints
X = hstack(
    [X_desc, X_fp],
    format="csr",
    dtype=np.float32
)

del X_desc, X_fp
gc.collect()

expected_features = 64 + 8192

if X.shape != (len(labels), expected_features):
    raise ValueError(
        f"Unexpected feature shape: {X.shape}"
    )

print("Combined feature matrix:", X.shape)
print("Feature count:", X.shape[1])

# ============================================================
# 10. CREATE TRAIN / VALIDATION / TEST MATRICES
# ============================================================

print("\n========== CREATING DATA SPLITS ==========")

X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

del X
gc.collect()

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# ============================================================
# 11. TRAIN RANDOM FOREST
# ============================================================

print("\n========== RANDOM FOREST TRAINING ==========")

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    max_features=MAX_FEATURES,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

training_start = time.time()

model.fit(
    X_train,
    y_train
)

training_time = time.time() - training_start

print("\nTraining completed.")
print("Training time:", round(training_time, 2), "seconds")

# ============================================================
# 12. VALIDATION EVALUATION
# ============================================================

print("\n========== VALIDATION RESULTS ==========")

val_predictions = model.predict(X_val)

val_metrics = {
    "accuracy": float(
        accuracy_score(y_val, val_predictions)
    ),
    "macro_precision": float(
        precision_score(
            y_val,
            val_predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )
    ),
    "macro_recall": float(
        recall_score(
            y_val,
            val_predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )
    ),
    "macro_f1": float(
        f1_score(
            y_val,
            val_predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )
    ),
    "weighted_f1": float(
        f1_score(
            y_val,
            val_predictions,
            average="weighted",
            zero_division=0
        )
    )
}

for key, value in val_metrics.items():
    print(f"{key}: {value:.4f}")

# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

print("\n========== FINAL TEST RESULTS ==========")

test_predictions = model.predict(X_test)

test_metrics = {
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "macro_precision": float(
        precision_score(
            y_test,
            test_predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )
    ),
    "macro_recall": float(
        recall_score(
            y_test,
            test_predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )
    ),
    "macro_f1": float(
        f1_score(
            y_test,
            test_predictions,
            labels=classes,
            average="macro",
            zero_division=0
        )
    ),
    "weighted_precision": float(
        precision_score(
            y_test,
            test_predictions,
            labels=classes,
            average="weighted",
            zero_division=0
        )
    ),
    "weighted_recall": float(
        recall_score(
            y_test,
            test_predictions,
            labels=classes,
            average="weighted",
            zero_division=0
        )
    ),
    "weighted_f1": float(
        f1_score(
            y_test,
            test_predictions,
            labels=classes,
            average="weighted",
            zero_division=0
        )
    ),
    "training_time_seconds": float(training_time),
    "total_records": int(len(labels)),
    "training_records": int(len(train_idx)),
    "validation_records": int(len(val_idx)),
    "test_records": int(len(test_idx)),
    "number_of_classes": int(len(classes)),
    "number_of_features": int(expected_features),
    "number_of_unique_pairs": int(len(np.unique(groups))),
    "number_of_estimators": int(N_ESTIMATORS),
    "number_of_splits": int(n_splits),
    "random_state": int(RANDOM_STATE)
}

for key, value in test_metrics.items():
    print(f"{key}: {value}")

# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_test,
    test_predictions,
    labels=classes,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

print(report_df.to_string())

# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

cm_df.to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)

print(cm_df)

# ============================================================
# 16. SAVE TEST PREDICTIONS
# ============================================================

print("\n========== SAVING TEST PREDICTIONS ==========")

predictions_df = labels.iloc[test_idx][
    [
        "Drug 1",
        "Drug 2",
        "canonical_pair_id",
        "interaction_family"
    ]
].copy()

predictions_df = predictions_df.rename(
    columns={
        "interaction_family": "actual_interaction_family"
    }
)

predictions_df["predicted_interaction_family"] = (
    test_predictions
)

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========")

descriptor_names = pd.read_csv(
    DESC_NAMES_FILE
)["feature_name"].astype(str).tolist()

fingerprint_names = pd.read_csv(
    FP_NAMES_FILE
)["feature_name"].astype(str).tolist()

feature_names = (
    descriptor_names + fingerprint_names
)

if len(feature_names) != expected_features:
    raise ValueError(
        f"Feature name count mismatch. "
        f"Expected {expected_features}, got {len(feature_names)}."
    )

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\nTop 15 features:")
print(
    importance.head(15).to_string(index=False)
)

# ============================================================
# 18. SAVE MODEL
# ============================================================

print("\n========== SAVING MODEL ==========")

model_path = (
    OUTPUT_DIR /
    "random_forest_interaction_family.pkl"
)

joblib.dump(
    model,
    model_path,
    compress=3
)

# Save evaluation metrics
metrics_output = {
    "validation": val_metrics,
    "test": test_metrics
}

with open(
    OUTPUT_DIR / "evaluation_metrics.json",
    "w"
) as f:
    json.dump(
        metrics_output,
        f,
        indent=4
    )

# Save model configuration and feature metadata
metadata = {
    "model": "RandomForestClassifier",
    "target": "interaction_family",
    "group_column": "canonical_pair_id",
    "feature_count": expected_features,
    "descriptor_count": 64,
    "fingerprint_count": 8192,
    "classes": classes.tolist(),
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "max_features": MAX_FEATURES,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_splits": n_splits,
    "split_allocation": {
        "training_folds": list(range(2, n_splits)),
        "validation_fold": 1,
        "test_fold": 0
    }
}

with open(
    OUTPUT_DIR / "model_metadata.json",
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )

# ============================================================
# 19. FINAL SUMMARY
# ============================================================

total_time = time.time() - start_time

print("\n==========================================")
print("       DDIvolve TRAINING COMPLETE")
print("==========================================")

print("Total records:", len(labels))
print("Unique drug pairs:", len(np.unique(groups)))
print("Interaction families:", len(classes))
print("Molecular features:", expected_features)

print("\nFinal test accuracy:",
      round(test_metrics["accuracy"], 4))

print("Final macro F1:",
      round(test_metrics["macro_f1"], 4))

print("Final weighted F1:",
      round(test_metrics["weighted_f1"], 4))

print("\nModel saved at:")
print(model_path)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nTotal execution time:",
      round(total_time, 2), "seconds")

print("\nAll pipeline stages completed successfully.")
