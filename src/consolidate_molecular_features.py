
from pathlib import Path
import numpy as np
import pandas as pd
import json

# ============================================================
# DDIvolve - Molecular Feature Consolidation
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
FEATURE_DIR = DATA_DIR / "molecular_features"

OUTPUT_DIR = DATA_DIR / "consolidated_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE = OUTPUT_DIR / "consolidated_molecular_features.csv"
REPORT_FILE = OUTPUT_DIR / "consolidation_report.txt"


# ------------------------------------------------------------
# 1. Utility functions
# ------------------------------------------------------------

def normalize_name(name):
    return str(name).strip().casefold()


def find_file(filename):
    matches = list(ROOT.rglob(filename))
    matches = [
        p for p in matches
        if OUTPUT_DIR not in p.parents
    ]

    if not matches:
        return None

    # Prefer files inside the expected feature directory.
    matches.sort(
        key=lambda p: (
            0 if FEATURE_DIR in p.parents else 1,
            len(p.parts)
        )
    )
    return matches[0]


def find_column(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def resolve_file(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return None

    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    if path.is_file():
        return path

    # Sometimes index files contain only the filename.
    matches = list(ROOT.rglob(path.name))

    for match in matches:
        if OUTPUT_DIR not in match.parents:
            return match

    return None


def load_index(filename, label):
    path = find_file(filename)

    if path is None:
        print(f"[WARNING] {label} index not found: {filename}")
        return {}

    df = pd.read_csv(path)

    drug_col = find_column(
        df,
        ["drug_name", "drug", "name", "canonical_name"]
    )

    if drug_col is None:
        print(f"[WARNING] No drug-name column in {path}")
        return {}

    file_columns = [
        c for c in df.columns
        if any(word in str(c).lower()
               for word in ["file", "path", "image", "structure"])
    ]

    result = {}

    for _, row in df.iterrows():
        name = normalize_name(row[drug_col])
        if not name or name == "nan":
            continue

        resolved = None

        for col in file_columns:
            resolved = resolve_file(row[col])
            if resolved:
                break

        # Preserve all index metadata.
        metadata = {
            str(k): v
            for k, v in row.to_dict().items()
            if pd.notna(v)
        }

        result[name] = {
            "path": str(resolved.relative_to(ROOT))
                    if resolved else "",
            "metadata": metadata
        }

    print(f"[OK] Loaded {len(result)} records from {label}")
    return result


# ------------------------------------------------------------
# 2. Load original drug dataset
# ------------------------------------------------------------

drug_file = DATA_DIR / "clean_drugs.csv"

if not drug_file.exists():
    raise FileNotFoundError(
        f"Original drug dataset not found: {drug_file}"
    )

drugs = pd.read_csv(drug_file)

drug_col = find_column(
    drugs,
    ["drug_name", "name"]
)

if drug_col is None:
    raise ValueError("Drug name column is missing.")

drugs["match_key"] = drugs[drug_col].apply(normalize_name)

if drugs["match_key"].duplicated().any():
    raise ValueError(
        "Duplicate drug names found in clean_drugs.csv. "
        "Resolve them before consolidation."
    )

print("\nOriginal drug records:", len(drugs))


# ------------------------------------------------------------
# 3. Load molecular descriptors
# ------------------------------------------------------------

descriptor_file = find_file("molecular_descriptors.csv")

if descriptor_file:
    descriptors = pd.read_csv(descriptor_file)

    descriptor_drug_col = find_column(
        descriptors,
        ["drug_name", "name"]
    )

    if descriptor_drug_col is not None:
        descriptors["match_key"] = (
            descriptors[descriptor_drug_col].apply(normalize_name)
        )

        # Keep only descriptor columns not already in the base data.
        descriptor_columns = [
            c for c in descriptors.columns
            if c not in [descriptor_drug_col, "match_key"]
            and c not in drugs.columns
        ]

        if descriptor_columns:
            drugs = drugs.merge(
                descriptors[["match_key"] + descriptor_columns],
                on="match_key",
                how="left",
                validate="one_to_one"
            )

        print("[OK] Separate descriptor file loaded.")
    else:
        print("[WARNING] Descriptor file has no drug-name column.")

else:
    print("[INFO] Using descriptors already present in clean_drugs.csv.")


# Identify descriptor columns.
descriptor_columns = [
    c for c in drugs.columns
    if c.lower() in {
        "molecular_formula",
        "molecular_weight",
        "mw",
        "logp",
        "h_bond_donors",
        "h_bond_acceptors",
        "tpsa",
        "rotatable_bonds",
        "qed",
        "atom_count",
        "bond_count"
    }
]

drugs["has_descriptors"] = (
    drugs[descriptor_columns].notna().all(axis=1)
    if descriptor_columns
    else False
)


# ------------------------------------------------------------
# 4. Load Morgan fingerprints
# ------------------------------------------------------------

fingerprint_file = find_file("morgan_fingerprints.npz")

fingerprint_map = {}

if fingerprint_file:

    data = np.load(fingerprint_file, allow_pickle=True)

    fingerprints = data["fingerprints"]
    fingerprint_names = data["drug_names"]

    if len(fingerprints) != len(fingerprint_names):
        raise ValueError(
            "Fingerprint count does not match drug-name count."
        )

    for i, name in enumerate(fingerprint_names):
        fingerprint_map[normalize_name(name)] = i

    fingerprint_dim = (
        fingerprints.shape[1]
        if fingerprints.ndim == 2 else 0
    )

    print(
        f"[OK] Morgan fingerprints: "
        f"{len(fingerprint_map)} drugs, "
        f"{fingerprint_dim} dimensions"
    )

else:
    print("[WARNING] Morgan fingerprint NPZ not found.")


# ------------------------------------------------------------
# 5. Load graph, 2D, 3D and optimized-3D indexes
# ------------------------------------------------------------

graph_map = load_index(
    "molecular_graph_index.csv",
    "Molecular graphs"
)

structure_2d_map = load_index(
    "2d_structure_index.csv",
    "2D structures"
)

conformer_3d_map = load_index(
    "3d_conformer_index.csv",
    "3D conformers"
)

optimized_3d_map = load_index(
    "optimization_index.csv",
    "Optimized 3D structures"
)


# ------------------------------------------------------------
# 6. Consolidate all feature availability information
# ------------------------------------------------------------

master = drugs.copy()

master["fingerprint_row"] = master["match_key"].map(
    fingerprint_map
)

master["has_fingerprint"] = (
    master["fingerprint_row"].notna()
)

master["graph_file"] = master["match_key"].map(
    lambda k: graph_map.get(k, {}).get("path", "")
)

master["has_graph"] = master["graph_file"].ne("")

master["structure_2d_file"] = master["match_key"].map(
    lambda k: structure_2d_map.get(k, {}).get("path", "")
)

master["has_2d_structure"] = (
    master["structure_2d_file"].ne("")
)

master["conformer_3d_file"] = master["match_key"].map(
    lambda k: conformer_3d_map.get(k, {}).get("path", "")
)

master["has_3d_conformer"] = (
    master["conformer_3d_file"].ne("")
)

master["optimized_3d_file"] = master["match_key"].map(
    lambda k: optimized_3d_map.get(k, {}).get("path", "")
)

master["has_optimized_3d"] = (
    master["optimized_3d_file"].ne("")
)


# ------------------------------------------------------------
# 7. Preserve optimization metadata
# ------------------------------------------------------------

def get_metadata(name, field):
    record = optimized_3d_map.get(name, {})
    metadata = record.get("metadata", {})

    for key, value in metadata.items():
        if key.lower() == field.lower():
            return value

    return None


for field in [
    "optimization_status",
    "status",
    "converged",
    "force_field",
    "final_energy",
    "energy"
]:
    master[field] = master["match_key"].apply(
        lambda name: get_metadata(name, field)
    )


# ------------------------------------------------------------
# 8. Calculate overall feature availability
# ------------------------------------------------------------

feature_flags = [
    "has_descriptors",
    "has_fingerprint",
    "has_graph",
    "has_2d_structure",
    "has_3d_conformer",
    "has_optimized_3d"
]

master["available_feature_count"] = (
    master[feature_flags].sum(axis=1)
)

master["all_features_available"] = (
    master[feature_flags].all(axis=1)
)

master["missing_features"] = master.apply(
    lambda row: ", ".join(
        flag.replace("has_", "")
        for flag in feature_flags
        if not bool(row[flag])
    ),
    axis=1
)

master.drop(columns=["match_key"], inplace=True)


# ------------------------------------------------------------
# 9. Save consolidated dataset
# ------------------------------------------------------------

master.to_csv(MASTER_FILE, index=False)


# ------------------------------------------------------------
# 10. Generate validation report
# ------------------------------------------------------------

lines = [
    "DDIvolve - Molecular Feature Consolidation Report",
    "=" * 55,
    f"Total original drug records: {len(master)}",
    "",
    "FEATURE AVAILABILITY",
    "-" * 35
]

for flag in feature_flags:
    count = int(master[flag].sum())
    percentage = count / len(master) * 100

    lines.append(
        f"{flag}: {count}/{len(master)} "
        f"({percentage:.2f}%)"
    )

lines.extend([
    "",
    f"All features available: "
    f"{int(master['all_features_available'].sum())}",
    f"Partial feature records: "
    f"{int((~master['all_features_available']).sum())}",
    "",
    "MISSING FEATURE DISTRIBUTION",
    "-" * 35
])

for flag in feature_flags:
    missing = int((~master[flag]).sum())
    lines.append(f"{flag}: {missing} missing")

lines.extend([
    "",
    f"Master dataset: {MASTER_FILE}",
    "Consolidation completed."
])

report = "\n".join(lines)

REPORT_FILE.write_text(report, encoding="utf-8")

print("\n" + report)

print("\n[COMPLETED]")
print("Master CSV:", MASTER_FILE)
print("Report:", REPORT_FILE)
