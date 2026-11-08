
from pathlib import Path
import pandas as pd

# Locate the evidence dataset
MODULE_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_FILE = MODULE_DIR / "data" / "processed" / "clinical_evidence.csv"

VALID_LEVELS = {"Minor", "Moderate", "Major", "Unknown"}


def normalize_key(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def make_pair_key(drug1_key, drug2_key):
    return tuple(sorted([
        normalize_key(drug1_key),
        normalize_key(drug2_key)
    ]))


# Load evidence once when the module is imported
_evidence_df = pd.read_csv(EVIDENCE_FILE, dtype=str).fillna("")

_evidence_lookup = {}

for _, row in _evidence_df.iterrows():
    pair = make_pair_key(row["drug1_key"], row["drug2_key"])
    if pair[0] and pair[1]:
        _evidence_lookup[pair] = row.to_dict()


def assess_severity(drug1_key, drug2_key):
    """
    Retrieve source-reported DDInter severity evidence.

    This is an evidence lookup, not an independently validated
    clinical severity prediction model.
    """

    pair = make_pair_key(drug1_key, drug2_key)

    if not pair[0] or not pair[1]:
        return {
            "matched": False,
            "severity": "Unknown",
            "assessment_status": "Invalid or missing drug key",
            "evidence_source": None,
            "message": "Please provide two valid drug keys."
        }

    record = _evidence_lookup.get(pair)

    if record is None:
        return {
            "matched": False,
            "severity": "Not assessed",
            "source_reported_level": None,
            "assessment_status": "Not assessed",
            "evidence_source": None,
            "clinical_effect": None,
            "clinical_significance": None,
            "management": None,
            "message": (
                "No exact matching pair was found in the imported "
                "DDInter evidence. This does not mean that no "
                "interaction exists."
            )
        }

    level = record.get("source_reported_level", "Unknown")

    if level not in VALID_LEVELS:
        level = "Unknown"

    return {
        "matched": True,
        "drug1_key": record["drug1_key"],
        "drug2_key": record["drug2_key"],
        "severity": level,
        "source_reported_level": level,
        "assessment_status": record["assessment_status"],
        "evidence_source": record["evidence_source"],
        "evidence_quality": record["evidence_quality"],
        "interaction_description": record["interaction_description"],
        "clinical_effect": record["clinical_effect"] or None,
        "clinical_significance": record["clinical_significance"] or None,
        "management": record["management"] or None,
        "evidence_scope": record["evidence_scope"],
        "message": (
            "Severity level retrieved from DDInter 2.0. "
            "Independent clinical verification is pending."
        )
    }
