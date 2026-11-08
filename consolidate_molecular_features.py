from pathlib import Path
import numpy as np
import pandas as pd


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
    """Normalize drug names so indexes can be matched consistently."""
    if pd.isna(name):
        return ""
    return str(name).strip().casefold()


def find_file(filename):
    """Find a file under the project root, preferring molecular_features."""
    matches = [
        p for p in ROOT.rglob(filename)
        if p.is_file() and OUTPUT_DIR not in p.parents
    ]

    if not matches:
        return None

    matches.sort(
        key=lambda p: (
            0 if FEATURE_DIR in p.parents else 1,
            len(p.parts)
        )
    )
    return matches[0]


def find_column(df, candidates):
    """Find a dataframe column using case-insensitive exact matching."""
    lookup = {str(c).strip().casefold(): c for c in df.columns}

    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]

    return None


def resolve_file(value):
    """
    Resolve an indexed file path.

    Supports absolute paths, project-root-relative paths, and indexes
    that contain only a filename. Returns a Path only if it is a file.
    """
    if pd.isna(value):
        return None

    value = str(value).strip().strip('"').strip("'")

    if not value or value.casefold() in {"nan", "none", "null"}:
        return None

    path = Path(value)

    # First try the path exactly as stored.
    if path.is_absolute():
        if path.is_file():
            return path
    else:
        project_path = ROOT / path
        if project_path.is_file():
            return project_path

    # Some index files store only the filename.
    filename = path.name
    if not filename:
        return None

    try:
        matches = ROOT.rglob(filename)
        for match in matches:
            if (
                match.is_file()
                and OUTPUT_DIR not in match.parents
            ):
                return match
    except OSError:
        return None

    return None


def path_relative_to_root(path):
    """Return a project-relative path when possible."""
    if path is None:
        return ""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        # Keep an absolute path if a valid file is outside ROOT.
        return str(path)


def load_index(filename, label):
    """
    Load an index and preserve:
      - one backward-compatible primary path
      - every individually resolved file path
      - a file-existence flag for every file column
      - all original index metadata

    In particular, this recognizes optimized_sdf and
    optimized_coordinates in optimization_index.csv.
    """
    index_path = find_file(filename)

    if index_path is None:
        print(f"[WARNING] {label} index not found: {filename}")
        return {}

    df = pd.read_csv(index_path)

    drug_col = find_column(
        df,
        ["drug_name", "drug", "name", "canonical_name"]
    )

    if drug_col is None:
        print(f"[WARNING] No drug-name column in {index_path}")
        return {}

    # Include both conventional path names and molecular-file
    # column names such as optimized_sdf / optimized_coordinates.
    file_keywords = [
        "file",
        "path",
        "image",
        "structure",
        "sdf",
        "coordinates",
        "npz"
    ]

    file_columns = [
        column
        for column in df.columns
        if any(
            keyword in str(column).strip().casefold()
            for keyword in file_keywords
        )
    ]

    result = {}

    for _, row in df.iterrows():
        name = normalize_name(row[drug_col])

        if not name:
            continue

        file_paths = {}
        file_exists = {}

        for column in file_columns:
            column_key = str(column).strip().casefold()
            resolved = resolve_file(row[column])

            file_paths[column_key] = path_relative_to_root(resolved)
            file_exists[column_key] = (
                resolved is not None and resolved.is_file()
            )

        # Keep the old "path" field so existing index consumers
        # continue to work.
        primary_path = next(
            (value for value in file_paths.values() if value),
            ""
        )

        # Preserve every non-empty source field, including
        # optimization method, convergence, status, and energy.
        metadata = {
            str(key): value
            for key, value in row.to_dict().items()
            if pd.notna(value)
        }

        result[name] = {
            "path": primary_path,
            "files": file_paths,
            "file_exists": file_exists,
            "metadata": metadata
        }

    print(
        f"[OK] Loaded {len(result)} records from {label} "
        f"({len(file_columns)} file columns detected)"
    )
    if file_columns:
        print(f"     File columns: {[str(c) for c in file_columns]}")

    return result


def get_index_path(index_map, name, preferred_columns=None):
    """Get a path from an index, preferring named file columns."""
    record = index_map.get(name, {})
    files = record.get("files", {})

    for column in preferred_columns or []:
        value = files.get(column.casefold(), "")
        if value:
            return value

    return record.get("path", "")


def get_index_file_exists(index_map, name, preferred_columns=None):
    """Return whether the selected indexed file physically exists."""
    record = index_map.get(name, {})
    file_exists = record.get("file_exists", {})

    for column in preferred_columns or []:
        key = column.casefold()
        if key in file_exists:
            return bool(file_exists[key])

    # Backward-compatible fallback for indexes with a single path.
    return bool(record.get("path", ""))


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

if drugs["match_key"].eq("").any():
    raise ValueError(
        "Blank drug names found in clean_drugs.csv. "
        "Resolve them before consolidation."
    )

if drugs["match_key"].duplicated().any():
    duplicates = drugs.loc[
        drugs["match_key"].duplicated(keep=False), drug_col
    ].astype(str).tolist()

    raise ValueError(
        "Duplicate drug names found in clean_drugs.csv. "
        f"Examples: {duplicates[:10]}. Resolve them before consolidation."
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

        # Avoid adding columns that already exist in the base dataset.
        descriptor_columns = [
            column for column in descriptors.columns
            if column not in [descriptor_drug_col, "match_key"]
            and column not in drugs.columns
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
    column for column in drugs.columns
    if str(column).strip().casefold() in {
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
fingerprint_dim = 0

if fingerprint_file:
    with np.load(fingerprint_file, allow_pickle=True) as data:
        if "fingerprints" not in data.files or "drug_names" not in data.files:
            raise ValueError(
                "Morgan fingerprint NPZ must contain "
                "'fingerprints' and 'drug_names' arrays."
            )

        fingerprints = data["fingerprints"]
        fingerprint_names = data["drug_names"]

        if len(fingerprints) != len(fingerprint_names):
            raise ValueError(
                "Fingerprint count does not match drug-name count."
            )

        for index, name in enumerate(fingerprint_names):
            normalized = normalize_name(name)
            if normalized:
                fingerprint_map[normalized] = index

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

master["has_fingerprint"] = master["fingerprint_row"].notna()


# Molecular graph files.
master["graph_file"] = master["match_key"].map(
    lambda key: get_index_path(graph_map, key)
)

master["has_graph"] = master["match_key"].map(
    lambda key: get_index_file_exists(graph_map, key)
)


# 2D structure files.
master["structure_2d_file"] = master["match_key"].map(
    lambda key: get_index_path(
        structure_2d_map,
        key,
        ["structure_2d_file", "2d_structure_file", "path"]
    )
)

master["has_2d_structure"] = master["match_key"].map(
    lambda key: get_index_file_exists(
        structure_2d_map,
        key,
        ["structure_2d_file", "2d_structure_file", "path"]
    )
)


# Original 3D conformer files.
master["conformer_3d_file"] = master["match_key"].map(
    lambda key: get_index_path(
        conformer_3d_map,
        key,
        ["conformer_3d_file", "3d_conformer_file", "path"]
    )
)

master["has_3d_conformer"] = master["match_key"].map(
    lambda key: get_index_file_exists(
        conformer_3d_map,
        key,
        ["conformer_3d_file", "3d_conformer_file", "path"]
    )
)


# Optimized SDF and coordinate files are stored separately.
master["optimized_sdf_file"] = master["match_key"].map(
    lambda key: get_index_path(
        optimized_3d_map,
        key,
        ["optimized_sdf", "optimized_sdf_file"]
    )
)

master["optimized_coordinates_file"] = master["match_key"].map(
    lambda key: get_index_path(
        optimized_3d_map,
        key,
        ["optimized_coordinates", "optimized_coordinates_file"]
    )
)

master["has_optimized_sdf"] = master["match_key"].map(
    lambda key: get_index_file_exists(
        optimized_3d_map,
        key,
        ["optimized_sdf", "optimized_sdf_file"]
    )
)

master["has_optimized_coordinates"] = master["match_key"].map(
    lambda key: get_index_file_exists(
        optimized_3d_map,
        key,
        ["optimized_coordinates", "optimized_coordinates_file"]
    )
)

# An optimized 3D record is complete only when BOTH files exist.
master["has_optimized_3d"] = (
    master["has_optimized_sdf"]
    & master["has_optimized_coordinates"]
)

# Retain the previous column name for compatibility with code
# that may already refer to optimized_3d_file.
master["optimized_3d_file"] = master["optimized_sdf_file"]


# ------------------------------------------------------------
# 7. Preserve optimization metadata
# ------------------------------------------------------------

def get_metadata(name, field):
    record = optimized_3d_map.get(name, {})
    metadata = record.get("metadata", {})

    for key, value in metadata.items():
        if str(key).strip().casefold() == field.casefold():
            return value

    return None


# Preserve the original fields and common method/convergence aliases.
metadata_fields = [
    "optimization_status",
    "status",
    "converged",
    "convergence_status",
    "force_field",
    "optimization_method",
    "method",
    "final_energy",
    "energy"
]

for field in metadata_fields:
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

master["available_feature_count"] = master[feature_flags].sum(axis=1)

master["all_features_available"] = master[feature_flags].all(axis=1)

master["missing_features"] = master.apply(
    lambda row: ", ".join(
        flag.replace("has_", "")
        for flag in feature_flags
        if not bool(row[flag])
    ),
    axis=1
)


# ------------------------------------------------------------
# 9. Save consolidated dataset
# ------------------------------------------------------------

master.drop(columns=["match_key"], inplace=True)

master.to_csv(MASTER_FILE, index=False)


# ------------------------------------------------------------
# 10. Generate validation report
# ------------------------------------------------------------

total_drugs = len(master)

lines = [
    "DDIvolve - Molecular Feature Consolidation Report",
    "=" * 55,
    f"Total original drug records: {total_drugs}",
    "",
    "FEATURE AVAILABILITY",
    "-" * 35
]

for flag in feature_flags:
    count = int(master[flag].sum())
    percentage = (count / total_drugs * 100) if total_drugs else 0

    lines.append(
        f"{flag}: {count}/{total_drugs} ({percentage:.2f}%)"
    )

lines.extend([
    "",
    "OPTIMIZED 3D FILE VALIDATION",
    "-" * 35,
    (
        f"Optimized SDF files present: "
        f"{int(master['has_optimized_sdf'].sum())}/{total_drugs}"
    ),
    (
        f"Optimized coordinate NPZ files present: "
        f"{int(master['has_optimized_coordinates'].sum())}/{total_drugs}"
    ),
    (
        f"Both optimized files present: "
        f"{int(master['has_optimized_3d'].sum())}/{total_drugs}"
    ),
    (
        f"SDF present but coordinates missing: "
        f"{int((master['has_optimized_sdf'] & ~master['has_optimized_coordinates']).sum())}"
    ),
    (
        f"Coordinates present but SDF missing: "
        f"{int((~master['has_optimized_sdf'] & master['has_optimized_coordinates']).sum())}"
    ),
    "",
    f"All features available: {int(master['all_features_available'].sum())}",
    f"Partial feature records: {int((~master['all_features_available']).sum())}",
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
