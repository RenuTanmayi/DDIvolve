
import os
import numpy as np
import pandas as pd
from rdkit import Chem

BASE = "data/processed/molecular_features"

INPUT_INDEX = os.path.join(
    BASE, "3d_conformers", "3d_conformer_index.csv"
)

OPT_DIR = os.path.join(BASE, "optimized_3d")
OPT_INDEX = os.path.join(OPT_DIR, "optimization_index.csv")

df_input = pd.read_csv(INPUT_INDEX)
df_opt = pd.read_csv(OPT_INDEX)

issues = []
validated = []

for _, row in df_opt.iterrows():
    name = row["drug_name"]
    sdf_path = os.path.join(OPT_DIR, "structures",
                            row["optimized_sdf"])
    npz_path = os.path.join(OPT_DIR, "coordinates",
                            row["optimized_coordinates"])

    status = "Valid"
    atom_count_sdf = None
    atom_count_npz = None

    # Check SDF
    if not os.path.exists(sdf_path):
        status = "Missing SDF"
    else:
        mol = Chem.MolFromMolFile(
            sdf_path, removeHs=False, sanitize=True
        )

        if mol is None or mol.GetNumConformers() == 0:
            status = "Invalid SDF"
        else:
            atom_count_sdf = mol.GetNumAtoms()

    # Check coordinate file
    if not os.path.exists(npz_path):
        status = "Missing coordinates"
    else:
        try:
            data = np.load(npz_path)
            if "coordinates" not in data:
                status = "Missing coordinates array"
            else:
                coords = data["coordinates"]
                atom_count_npz = coords.shape[0]

                if coords.ndim != 2 or coords.shape[1] != 3:
                    status = "Invalid coordinate shape"
                elif not np.isfinite(coords).all():
                    status = "Non-finite coordinates"
                elif (atom_count_sdf is not None
                      and atom_count_sdf != atom_count_npz):
                    status = "Atom count mismatch"
        except Exception as e:
            status = f"NPZ error: {e}"

    # Check final energy
    try:
        energy = float(row["final_energy"])
        if not np.isfinite(energy):
            status = "Invalid final energy"
    except (ValueError, TypeError, KeyError):
        status = "Missing/invalid final energy"

    validated.append({
        "drug_name": name,
        "validation_status": status,
        "optimization_method": row["optimization_method"],
        "optimization_status": row["optimization_status"],
        "atoms_in_sdf": atom_count_sdf,
        "atoms_in_npz": atom_count_npz
    })

    if status != "Valid":
        issues.append((name, status))

report = pd.DataFrame(validated)

# Compare input and optimized drug names
input_names = set(df_input["drug_name"])
optimized_names = set(df_opt["drug_name"])

failed_names = sorted(input_names - optimized_names)

# Save audit report
report.to_csv(
    os.path.join(OPT_DIR, "validation_report.csv"),
    index=False
)

print("\n========== VALIDATION SUMMARY ==========")
print("Input conformers:", len(df_input))
print("Optimized index records:", len(df_opt))
print("Valid structures:", (report["validation_status"] == "Valid").sum())
print("Structures with issues:", len(issues))
print("Optimization failures:", len(failed_names))

print("\nValidation status:")
print(report["validation_status"].value_counts())

print("\nOptimization methods:")
print(report["optimization_method"].value_counts())

print("\nConvergence status:")
print(report["optimization_status"].value_counts())

print("\nFailed optimization drugs:")
print(failed_names)

print("\nNon-converged drugs:")
print(
    report.loc[
        report["optimization_status"] == "Not converged",
        "drug_name"
    ].tolist()
)

print("\nUFF structures requiring review:")
print(
    report.loc[
        report["optimization_method"] == "UFF",
        "drug_name"
    ].tolist()
)

print("\nValidation issues:")
for issue in issues:
    print(issue)

print("\nReport saved to:", os.path.join(
    OPT_DIR, "validation_report.csv"
))
