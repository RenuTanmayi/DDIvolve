"""
DDIvolve Phase 1: Data cleaning, matching, integration, and quality reporting.

Expected inputs:
    data/interactions.csv
    data/drugs.csv

Outputs:
    data/raw/interactions.csv
    data/raw/drugs.csv
    data/processed/clean_interactions.csv
    data/processed/clean_drugs.csv
    data/processed/integrated_ddi_dataset.csv
    data/reports/data_quality_report.csv

Run from the project folder:
    python prepare_ddi_data.py

Dependencies:
    pip install pandas
    RDKit is optional. If installed, SMILES are validated with RDKit.
"""

from pathlib import Path
import shutil
import re
import unicodedata
import pandas as pd

# RDKit is optional. The script still runs without it.
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDKIT_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

INPUT_CANDIDATES = {
    "interactions": [
        RAW_DIR / "interactions.csv",
        DATA_DIR / "interactions.csv",
    ],
    "drugs": [
        RAW_DIR / "drugs.csv",
        DATA_DIR / "drugs.csv",
    ],
}


def make_directories():
    for folder in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def find_and_archive_input(kind):
    """Find an input CSV and copy it into data/raw without deleting the source."""
    for path in INPUT_CANDIDATES[kind]:
        if path.exists():
            destination = RAW_DIR / f"{kind}.csv"
            if path.resolve() != destination.resolve():
                # Keep the original file in place as a safety copy.
                shutil.copy2(path, destination)
            return destination
    searched = "\n".join(str(p) for p in INPUT_CANDIDATES[kind])
    raise FileNotFoundError(
        f"Could not find {kind}.csv. Checked:\n{searched}"
    )


def read_csv_safely(path):
    """Read CSV with common encodings and preserve drug names as text."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode CSV: {path}")


def clean_name(value):
    """Conservative name normalization; does not remove meaningful punctuation."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def name_key(value):
    """Case-insensitive matching key while retaining a readable display name."""
    return clean_name(value).casefold()


def add_report(rows, metric, value, details=""):
    rows.append({
        "metric": metric,
        "value": value,
        "details": details
    })


def clean_interaction_data(interactions):
    if interactions.shape[1] < 2:
        raise ValueError(
            "interactions.csv must contain at least two columns for the drug pair."
        )

    original_columns = list(interactions.columns)
    drug_a_col, drug_b_col = original_columns[0], original_columns[1]

    # Retain the third column as supplied, but do not assume it is a
    # human-readable clinical description. Preserve all other source columns.
    cleaned = interactions.copy()
    cleaned[drug_a_col] = cleaned[drug_a_col].map(clean_name)
    cleaned[drug_b_col] = cleaned[drug_b_col].map(clean_name)

    before = len(cleaned)
    cleaned = cleaned[
        (cleaned[drug_a_col] != "") &
        (cleaned[drug_b_col] != "")
    ].copy()
    missing_pair_rows = before - len(cleaned)

    cleaned["drug1"] = cleaned[drug_a_col]
    cleaned["drug2"] = cleaned[drug_b_col]
    cleaned["drug1_key"] = cleaned["drug1"].map(name_key)
    cleaned["drug2_key"] = cleaned["drug2"].map(name_key)

    # Drug-drug pairs are treated as unordered for duplicate detection.
    # Original drug1/drug2 order is retained in the output.
    pair_keys = cleaned.apply(
        lambda row: tuple(sorted((row["drug1_key"], row["drug2_key"]))),
        axis=1
    )
    cleaned["_pair_key"] = pair_keys

    # Include all non-drug-pair source fields in duplicate detection so that
    # separate interaction labels/descriptions for the same pair are retained.
    source_value_cols = [
        c for c in original_columns
        if c not in (drug_a_col, drug_b_col)
    ]
    if source_value_cols:
        cleaned["_source_signature"] = cleaned[source_value_cols].fillna("").astype(str).agg(
            " | ".join, axis=1
        )
    else:
        cleaned["_source_signature"] = ""

    before_duplicates = len(cleaned)
    cleaned = cleaned.drop_duplicates(
        subset=["_pair_key", "_source_signature"]
    ).copy()
    duplicate_rows = before_duplicates - len(cleaned)

    cleaned["pair_id"] = cleaned.apply(
        lambda row: " || ".join(sorted((row["drug1_key"], row["drug2_key"]))),
        axis=1
    )

    cleaned.drop(
        columns=["_pair_key", "_source_signature"],
        inplace=True,
        errors="ignore"
    )
    cleaned.reset_index(drop=True, inplace=True)

    stats = {
        "missing_drug_pair_rows_removed": missing_pair_rows,
        "duplicate_interaction_rows_removed": duplicate_rows,
        "interaction_rows_after_cleaning": len(cleaned),
        "interaction_drug1_column_used": drug_a_col,
        "interaction_drug2_column_used": drug_b_col,
        "interaction_description_or_label_column": (
            original_columns[2] if len(original_columns) >= 3 else "Not available"
        ),
    }
    return cleaned, stats


def clean_drug_data(drugs):
    if "drug_name" not in drugs.columns:
        raise ValueError(
            "drugs.csv must contain a 'drug_name' column. "
            f"Columns found: {list(drugs.columns)}"
        )

    cleaned = drugs.copy()
    cleaned["drug_name"] = cleaned["drug_name"].map(clean_name)
    before = len(cleaned)
    cleaned = cleaned[cleaned["drug_name"] != ""].copy()
    blank_names_removed = before - len(cleaned)

    cleaned["drug_key"] = cleaned["drug_name"].map(name_key)

    # Prefer a successful record with a non-empty SMILES when duplicate
    # normalized names occur; otherwise retain the first available record.
    if "lookup_status" in cleaned.columns:
        status = cleaned["lookup_status"].fillna("").astype(str).str.casefold()
        has_smiles = (
            cleaned["smiles"].fillna("").astype(str).str.strip().ne("")
            if "smiles" in cleaned.columns
            else pd.Series(False, index=cleaned.index)
        )
        cleaned["_record_priority"] = (
            status.eq("success").astype(int) * 2 +
            has_smiles.astype(int)
        )
        cleaned = cleaned.sort_values(
            "_record_priority", ascending=False, kind="stable"
        )

    before_duplicates = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["drug_key"], keep="first").copy()
    duplicate_drug_rows_removed = before_duplicates - len(cleaned)
    cleaned.drop(columns=["_record_priority"], inplace=True, errors="ignore")

    if "smiles" not in cleaned.columns:
        cleaned["smiles"] = ""

    cleaned["smiles"] = cleaned["smiles"].fillna("").astype(str).str.strip()

    if RDKIT_AVAILABLE:
        cleaned["smiles_validated"] = cleaned["smiles"].map(
            lambda s: bool(s) and Chem.MolFromSmiles(s) is not None
        )
    else:
        # Do not claim chemical validation when RDKit is unavailable.
        cleaned["smiles_validated"] = pd.NA

    if "pubchem_cid" in cleaned.columns:
        cleaned["pubchem_cid"] = cleaned["pubchem_cid"].fillna("").astype(str).str.strip()

    stats = {
        "drug_rows_in_source": before,
        "blank_drug_names_removed": blank_names_removed,
        "duplicate_normalized_drug_rows_removed": duplicate_drug_rows_removed,
        "unique_drug_records_after_cleaning": len(cleaned),
        "rows_with_smiles": int(cleaned["smiles"].ne("").sum()),
        "valid_smiles_count": (
            int(cleaned["smiles_validated"].fillna(False).sum())
            if RDKIT_AVAILABLE else "Not checked - RDKit not installed"
        ),
        "rdkit_available": RDKIT_AVAILABLE,
    }

    cleaned.reset_index(drop=True, inplace=True)
    return cleaned, stats


def integrate_datasets(interactions, drugs):
    """Join molecular information for both members of each interaction pair."""
    drug_lookup = drugs.drop_duplicates("drug_key").set_index("drug_key")

    interaction_keys = set(interactions["drug1_key"]) | set(interactions["drug2_key"])
    available_keys = set(drug_lookup.index)
    unmatched_keys = sorted(interaction_keys - available_keys)

    drug_columns = [
        c for c in drugs.columns
        if c not in ("drug_name", "drug_key")
    ]

    left = interactions.copy()
    right = interactions.copy()

    left_lookup = drug_lookup[drug_columns].add_prefix("drug1_")
    right_lookup = drug_lookup[drug_columns].add_prefix("drug2_")

    left = left.join(left_lookup, on="drug1_key")
    right = right.join(right_lookup, on="drug2_key")

    # The two joins have the same interaction rows and index. Keep the
    # drug1-prefixed molecular columns from the first join and drug2 from second.
    drug1_cols = [c for c in left.columns if c.startswith("drug1_") and c != "drug1_key"]
    drug2_cols = [c for c in right.columns if c.startswith("drug2_") and c != "drug2_key"]

    base_cols = list(interactions.columns)
    integrated = left[base_cols + drug1_cols].copy()
    for col in drug2_cols:
        integrated[col] = right[col]

    # Determine matches from normalized keys, not from whether descriptor
    # columns happen to contain non-null values.
    integrated["drug1_molecular_record_found"] = integrated["drug1_key"].isin(available_keys)
    integrated["drug2_molecular_record_found"] = integrated["drug2_key"].isin(available_keys)

    unmatched_pair_rows = int(
        (~integrated["drug1_molecular_record_found"] |
         ~integrated["drug2_molecular_record_found"]).sum()
    )

    stats = {
        "unique_interaction_drugs": len(interaction_keys),
        "drug_names_without_molecular_record": len(unmatched_keys),
        "interaction_rows_with_at_least_one_unmatched_drug": unmatched_pair_rows,
        "unmatched_drug_names": "; ".join(unmatched_keys[:100]),
    }
    return integrated, stats


def build_quality_report(interactions_raw, drugs_raw, clean_interactions,
                         clean_drugs, integrated, interaction_stats,
                         drug_stats, integration_stats):
    report = []

    add_report(report, "raw_interaction_rows", len(interactions_raw))
    add_report(report, "raw_interaction_columns", len(interactions_raw.columns))
    add_report(report, "raw_drug_rows", len(drugs_raw))
    add_report(report, "raw_drug_columns", len(drugs_raw.columns))

    for key, value in interaction_stats.items():
        add_report(report, key, value)

    for key, value in drug_stats.items():
        add_report(report, key, value)

    for key, value in integration_stats.items():
        add_report(report, key, value)

    add_report(report, "integrated_dataset_rows", len(integrated))
    add_report(report, "integrated_dataset_columns", len(integrated.columns))
    add_report(
        report,
        "integrated_rows_with_both_molecular_records",
        int((
            integrated["drug1_molecular_record_found"] &
            integrated["drug2_molecular_record_found"]
        ).sum())
    )

    # Missing-value counts for key fields.
    for frame_name, frame, columns in (
        ("clean_interactions", clean_interactions,
         ["drug1", "drug2"]),
        ("clean_drugs", clean_drugs,
         ["drug_name", "smiles"]),
        ("integrated_dataset", integrated,
         ["drug1", "drug2", "drug1_smiles", "drug2_smiles"]),
    ):
        for col in columns:
            if col in frame.columns:
                missing = frame[col].isna() | frame[col].astype(str).str.strip().eq("")
                add_report(
                    report,
                    f"{frame_name}_missing_{col}",
                    int(missing.sum())
                )

    return pd.DataFrame(report)


def main():
    make_directories()

    interactions_path = find_and_archive_input("interactions")
    drugs_path = find_and_archive_input("drugs")

    print("Reading source datasets...")
    interactions_raw = read_csv_safely(interactions_path)
    drugs_raw = read_csv_safely(drugs_path)

    print("Cleaning interaction data...")
    clean_interactions, interaction_stats = clean_interaction_data(
        interactions_raw
    )

    print("Cleaning molecular drug data...")
    clean_drugs, drug_stats = clean_drug_data(drugs_raw)

    print("Matching and integrating datasets...")
    integrated, integration_stats = integrate_datasets(
        clean_interactions, clean_drugs
    )

    print("Generating data quality report...")
    report = build_quality_report(
        interactions_raw,
        drugs_raw,
        clean_interactions,
        clean_drugs,
        integrated,
        interaction_stats,
        drug_stats,
        integration_stats,
    )

    clean_interactions.to_csv(
        PROCESSED_DIR / "clean_interactions.csv", index=False
    )
    clean_drugs.to_csv(
        PROCESSED_DIR / "clean_drugs.csv", index=False
    )
    integrated.to_csv(
        PROCESSED_DIR / "integrated_ddi_dataset.csv", index=False
    )
    report.to_csv(
        REPORTS_DIR / "data_quality_report.csv", index=False
    )

    print("\nPHASE 1 COMPLETED")
    print(f"Clean interactions: {len(clean_interactions)}")
    print(f"Clean drug records: {len(clean_drugs)}")
    print(f"Integrated interaction rows: {len(integrated)}")
    print(
        "Drug names without molecular records: "
        f"{integration_stats['drug_names_without_molecular_record']}"
    )
    print("\nGenerated files:")
    print(PROCESSED_DIR / "clean_interactions.csv")
    print(PROCESSED_DIR / "clean_drugs.csv")
    print(PROCESSED_DIR / "integrated_ddi_dataset.csv")
    print(REPORTS_DIR / "data_quality_report.csv")
    print("\nRaw source copies are stored in:")
    print(RAW_DIR)


if __name__ == "__main__":
    main()
