
import pandas as pd
from pathlib import Path

# ============================================================
# DDIVOLVE — STAGE A: SEVERITY DATASET AUDIT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "shared"

OUTPUT_DIR = BASE_DIR / "severity_module" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "drugs": DATA_DIR / "clean_drugs.csv",
    "templates": DATA_DIR / "interaction_template_catalog.csv",
    "labels": DATA_DIR / "validated_interaction_labels.csv",
    "pairs": DATA_DIR / "validated_pair_summary.csv"
}

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} dataset: {path}")

print("=" * 75)
print("DDIVOLVE — SEVERITY DATASET AUDIT")
print("=" * 75)

# Load datasets
drugs = pd.read_csv(FILES["drugs"], low_memory=False)
templates = pd.read_csv(FILES["templates"], low_memory=False)
labels = pd.read_csv(FILES["labels"], low_memory=False)
pairs = pd.read_csv(FILES["pairs"], low_memory=False)

audit_rows = []


def add_metric(section, metric, value, notes=""):
    audit_rows.append({
        "section": section,
        "metric": metric,
        "value": value,
        "notes": notes
    })


# ------------------------------------------------------------
# 1. BASIC DATASET STATISTICS
# ------------------------------------------------------------

print("\n[1] BASIC STATISTICS")

add_metric("Dataset", "Total drug records", len(drugs))
add_metric("Dataset", "Total interaction templates", len(templates))
add_metric("Dataset", "Total interaction label records", len(labels))
add_metric("Dataset", "Total pair summary records", len(pairs))

add_metric(
    "Dataset",
    "Unique drugs in clean_drugs",
    drugs["drug_key"].nunique()
)

add_metric(
    "Dataset",
    "Unique canonical drug pairs in labels",
    labels["canonical_pair_id"].nunique()
)

add_metric(
    "Dataset",
    "Unique canonical drug pairs in summary",
    pairs["canonical_pair_id"].nunique()
)

# ------------------------------------------------------------
# 2. INTERACTION FAMILY ANALYSIS
# ------------------------------------------------------------

print("\n[2] INTERACTION FAMILIES")

family_counts = (
    labels.groupby("interaction_family")
    .agg(
        record_count=("interaction_family", "size"),
        unique_pairs=("canonical_pair_id", "nunique")
    )
    .sort_values("record_count", ascending=False)
)

add_metric(
    "Interaction Families",
    "Number of unique interaction families",
    labels["interaction_family"].nunique()
)

for family, row in family_counts.iterrows():
    add_metric(
        "Family Frequency",
        str(family),
        int(row["record_count"]),
        f'Unique pairs: {row["unique_pairs"]}'
    )

print(family_counts.to_string())

# ------------------------------------------------------------
# 3. INTERACTION DESCRIPTION ANALYSIS
# ------------------------------------------------------------

print("\n[3] MOST FREQUENT INTERACTION DESCRIPTIONS")

description_counts = (
    labels.groupby("Interaction Description")
    .agg(
        record_count=("Interaction Description", "size"),
        unique_pairs=("canonical_pair_id", "nunique")
    )
    .sort_values("record_count", ascending=False)
)

for description, row in description_counts.head(20).iterrows():
    add_metric(
        "Top Interaction Descriptions",
        str(description),
        int(row["record_count"]),
        f'Unique pairs: {row["unique_pairs"]}'
    )

print(description_counts.head(20).to_string())

# ------------------------------------------------------------
# 4. MULTIPLE INTERACTION DESCRIPTIONS PER PAIR
# ------------------------------------------------------------

print("\n[4] MULTIPLE DESCRIPTIONS PER PAIR")

pair_analysis = (
    labels.groupby("canonical_pair_id")
    .agg(
        interaction_records=("Interaction Description", "size"),
        unique_descriptions=("Interaction Description", "nunique"),
        unique_families=("interaction_family", "nunique"),
        unique_affected_drugs=("affected_drug_key", "nunique")
    )
)

multiple_descriptions = pair_analysis[
    pair_analysis["unique_descriptions"] > 1
]

multiple_families = pair_analysis[
    pair_analysis["unique_families"] > 1
]

multiple_affected = pair_analysis[
    pair_analysis["unique_affected_drugs"] > 1
]

add_metric(
    "Pair Analysis",
    "Pairs with multiple interaction descriptions",
    len(multiple_descriptions)
)

add_metric(
    "Pair Analysis",
    "Pairs with multiple interaction families",
    len(multiple_families)
)

add_metric(
    "Pair Analysis",
    "Pairs with multiple affected-drug records",
    len(multiple_affected)
)

add_metric(
    "Pair Analysis",
    "Pairs with more than one interaction record",
    int((pair_analysis["interaction_records"] > 1).sum())
)

# ------------------------------------------------------------
# 5. VALIDATE PAIR SUMMARY AGAINST LABELS
# ------------------------------------------------------------

print("\n[5] PAIR SUMMARY VALIDATION")

label_pair_ids = set(labels["canonical_pair_id"].dropna())
summary_pair_ids = set(pairs["canonical_pair_id"].dropna())

missing_from_summary = label_pair_ids - summary_pair_ids
missing_from_labels = summary_pair_ids - label_pair_ids

summary_comparison = pairs.merge(
    pair_analysis.reset_index(),
    on="canonical_pair_id",
    how="left",
    suffixes=("_summary", "_labels")
)

interaction_mismatches = (
    summary_comparison["interaction_count"]
    != summary_comparison["interaction_records"]
).sum()

family_mismatches = (
    summary_comparison["unique_families_summary"]
    != summary_comparison["unique_families_labels"]
).sum()

affected_mismatches = (
    summary_comparison["unique_affected_targets"]
    != summary_comparison["unique_affected_drugs"]
).sum()

add_metric(
    "Consistency",
    "Label pairs absent from pair summary",
    len(missing_from_summary)
)

add_metric(
    "Consistency",
    "Summary pairs absent from labels",
    len(missing_from_labels)
)

add_metric(
    "Consistency",
    "Interaction count mismatches",
    int(interaction_mismatches)
)

add_metric(
    "Consistency",
    "Family count mismatches",
    int(family_mismatches)
)

add_metric(
    "Consistency",
    "Affected target count mismatches",
    int(affected_mismatches)
)

# ------------------------------------------------------------
# 6. DATA QUALITY
# ------------------------------------------------------------

print("\n[6] DATA QUALITY")

for name, df in [
    ("Drugs", drugs),
    ("Templates", templates),
    ("Labels", labels),
    ("Pair Summary", pairs)
]:
    add_metric(
        "Data Quality",
        f"{name}: duplicate rows",
        int(df.duplicated().sum())
    )

    add_metric(
        "Data Quality",
        f"{name}: total missing cells",
        int(df.isna().sum().sum())
    )

for column in ["smiles", "canonical_smiles", "drug_key"]:
    if column in drugs.columns:
        add_metric(
            "Drug Data Quality",
            f"Missing {column}",
            int(drugs[column].isna().sum())
        )

add_metric(
    "Drug Data Quality",
    "Drugs with valid SMILES",
    int(drugs["valid_smiles"].fillna(False).sum())
)

add_metric(
    "Drug Data Quality",
    "Drugs with missing molecular weight",
    int(drugs["molecular_weight"].isna().sum())
)

# ------------------------------------------------------------
# 7. CLINICAL EVIDENCE READINESS
# ------------------------------------------------------------

print("\n[7] CLINICAL EVIDENCE READINESS")

clinical_fields = [
    "clinical_effect",
    "severity",
    "clinical_significance",
    "management",
    "evidence_source",
    "evidence_quality"
]

available_fields = [
    field for field in clinical_fields
    if field in labels.columns
]

missing_fields = [
    field for field in clinical_fields
    if field not in labels.columns
]

add_metric(
    "Clinical Evidence",
    "Clinical evidence fields available",
    len(available_fields),
    ", ".join(available_fields)
)

add_metric(
    "Clinical Evidence",
    "Clinical evidence fields absent",
    len(missing_fields),
    ", ".join(missing_fields)
)

add_metric(
    "Clinical Evidence",
    "Pairs with verified severity in shared datasets",
    0,
    "No verified clinical severity labels are supplied in these four files."
)

add_metric(
    "Clinical Evidence",
    "Pairs requiring clinical evidence assessment",
    len(label_pair_ids),
    "Requires evidence retrieval and independent assessment."
)

add_metric(
    "Clinical Evidence",
    "Pairs with insufficient evidence",
    "Not yet determined",
    "Must be determined after clinical evidence collection."
)

# ------------------------------------------------------------
# 8. EXPORT AUDIT
# ------------------------------------------------------------

audit_df = pd.DataFrame(audit_rows)

audit_path = OUTPUT_DIR / "severity_dataset_audit.csv"
audit_df.to_csv(audit_path, index=False)

# ------------------------------------------------------------
# 9. EXPORT DATASET DOCUMENTATION
# ------------------------------------------------------------

documentation = f"""# DDIvolve — Severity Dataset Documentation

## 1. Purpose

This document describes the shared datasets used to develop
the DDIvolve Clinical Severity and Evidence Assessment Module.

## 2. Source files

| Dataset | Records | Purpose |
|---|---:|---|
| clean_drugs.csv | {len(drugs):,} | Drug identities and molecular information |
| interaction_template_catalog.csv | {len(templates):,} | Interaction template and family mappings |
| validated_interaction_labels.csv | {len(labels):,} | Validated interaction descriptions and labels |
| validated_pair_summary.csv | {len(pairs):,} | Pair-level interaction statistics |

## 3. Dataset statistics

- Unique drug records: {drugs['drug_key'].nunique():,}
- Unique drug pairs in labels: {labels['canonical_pair_id'].nunique():,}
- Unique pairs in summary: {pairs['canonical_pair_id'].nunique():,}
- Interaction families: {labels['interaction_family'].nunique():,}
- Interaction descriptions: {labels['Interaction Description'].nunique():,}

## 4. Pair-level structure

- Pairs with multiple descriptions: {len(multiple_descriptions):,}
- Pairs with multiple families: {len(multiple_families):,}
- Pairs with multiple affected-drug records: {len(multiple_affected):,}

## 5. Data consistency

- Label pairs missing from summary: {len(missing_from_summary):,}
- Summary pairs missing from labels: {len(missing_from_labels):,}
- Interaction count mismatches: {interaction_mismatches:,}
- Family count mismatches: {family_mismatches:,}
- Affected target count mismatches: {affected_mismatches:,}

## 6. Clinical evidence limitations

The shared interaction labels describe interaction mechanisms,
families, directions and affected drugs.

They do not establish clinical severity.

An interaction family must not be treated as a Minor, Moderate,
Major or Contraindicated severity label.

Clinical severity requires additional evidence from reliable
clinical sources.

Absence of evidence must not be interpreted as evidence of
absence of an interaction.

## 7. Severity assessment categories

- Minor
- Moderate
- Major
- Contraindicated
- Unknown

Evidence quality must be recorded independently from severity.

## 8. Next development stage

Collect pair-specific clinical evidence, preserve source references,
document management information, and independently validate
severity assessments before building the severity engine.

## 9. Important limitations

This audit assesses only the four shared datasets.
It does not establish that no clinical evidence exists elsewhere.
All clinical severity assessments remain pending until evidence
collection and validation are completed.
"""

documentation_path = OUTPUT_DIR / "DATASET_DOCUMENTATION.md"
documentation_path.write_text(documentation, encoding="utf-8")

# ------------------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("STAGE A AUDIT COMPLETED")
print("=" * 75)

print(f"\nUnique drug pairs: {len(label_pair_ids):,}")
print(f"Interaction families: {labels['interaction_family'].nunique():,}")
print(f"Multiple descriptions: {len(multiple_descriptions):,}")
print(f"Multiple affected-drug records: {len(multiple_affected):,}")

print("\nGENERATED FILES:")
print(audit_path)
print(documentation_path)

print("\nNo original dataset files were modified.")
print("=" * 75)
