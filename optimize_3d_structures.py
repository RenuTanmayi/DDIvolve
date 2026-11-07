
import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem

# ==========================================
# PATHS
# ==========================================

INPUT_DIR = "data/processed/molecular_features/3d_conformers"

INDEX_FILE = os.path.join(
    INPUT_DIR,
    "3d_conformer_index.csv"
)

OUTPUT_DIR = "data/processed/molecular_features/optimized_3d"

SDF_DIR = os.path.join(OUTPUT_DIR, "structures")
COORD_DIR = os.path.join(OUTPUT_DIR, "coordinates")

os.makedirs(SDF_DIR, exist_ok=True)
os.makedirs(COORD_DIR, exist_ok=True)

# ==========================================
# LOAD CONFORMER INDEX
# ==========================================

df = pd.read_csv(INDEX_FILE)

successful = 0
failed = []
not_converged = []

optimization_index = []

print("Starting 3D geometry optimization...\n")

# ==========================================
# OPTIMIZE MOLECULAR STRUCTURES
# ==========================================

for _, row in df.iterrows():

    drug_name = row["drug_name"]
    sdf_file = row["sdf_file"]
    coord_file = row["coordinate_file"]

    input_path = os.path.join(
        INPUT_DIR, "structures", sdf_file
    )

    mol = Chem.MolFromMolFile(
        input_path,
        removeHs=False,
        sanitize=True
    )

    if mol is None or mol.GetNumConformers() == 0:
        failed.append(drug_name)
        continue

    try:

        method = None
        status = None
        energy = None

        # ----------------------------------
        # Try MMFF94s first
        # ----------------------------------

        if AllChem.MMFFHasAllMoleculeParams(mol):

            properties = AllChem.MMFFGetMoleculeProperties(
                mol,
                mmffVariant="MMFF94s"
            )

            if properties is not None:

                result = AllChem.MMFFOptimizeMolecule(
                    mol,
                    mmffVariant="MMFF94s",
                    maxIters=1000
                )

                forcefield = AllChem.MMFFGetMoleculeForceField(
                    mol,
                    properties
                )

                if forcefield is not None:
                    energy = forcefield.CalcEnergy()
                    method = "MMFF94s"
                    status = (
                        "Converged" if result == 0
                        else "Not converged"
                    )

        # ----------------------------------
        # Fallback to UFF
        # ----------------------------------

        if method is None:

            if AllChem.UFFHasAllMoleculeParams(mol):

                result = AllChem.UFFOptimizeMolecule(
                    mol,
                    maxIters=1000
                )

                forcefield = AllChem.UFFGetMoleculeForceField(
                    mol
                )

                if forcefield is not None:
                    energy = forcefield.CalcEnergy()
                    method = "UFF"
                    status = (
                        "Converged" if result == 0
                        else "Not converged"
                    )

        # ----------------------------------
        # Handle unsupported molecules
        # ----------------------------------

        if method is None or energy is None:
            failed.append(drug_name)
            continue

        # ----------------------------------
        # Save optimized structure
        # ----------------------------------

        optimized_sdf = f"optimized_{sdf_file}"
        optimized_coord = f"optimized_{coord_file}"

        sdf_path = os.path.join(
            SDF_DIR, optimized_sdf
        )

        coord_path = os.path.join(
            COORD_DIR, optimized_coord
        )

        Chem.MolToMolFile(mol, sdf_path)

        # Extract optimized 3D coordinates
        coordinates = np.array(
            mol.GetConformer().GetPositions(),
            dtype=np.float32
        )

        np.savez_compressed(
            coord_path,
            drug_name=np.array(drug_name),
            coordinates=coordinates,
            energy=np.array(energy),
            optimization_method=np.array(method),
            optimization_status=np.array(status)
        )

        if status == "Not converged":
            not_converged.append(drug_name)

        optimization_index.append({

            "drug_name": drug_name,
            "optimized_sdf": optimized_sdf,
            "optimized_coordinates": optimized_coord,
            "optimization_method": method,
            "optimization_status": status,
            "final_energy": energy

        })

        successful += 1

    except Exception as e:

        failed.append(drug_name)
        print(f"Error: {drug_name}: {e}")

# ==========================================
# SAVE RESULTS
# ==========================================

result_df = pd.DataFrame(optimization_index)

result_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "optimization_index.csv"
    ),
    index=False
)

# ==========================================
# FINAL REPORT
# ==========================================

print("\n========== OPTIMIZATION RESULTS ==========")

print("Input conformers:", len(df))
print("Successfully optimized:", successful)
print("Failed:", len(failed))
print("Not converged:", len(not_converged))

if successful > 0:

    print("\nOptimization methods:")

    print(
        result_df["optimization_method"].value_counts()
    )

    print("\nConvergence status:")

    print(
        result_df["optimization_status"].value_counts()
    )

    print("\nAverage final energy:")

    print(result_df["final_energy"].mean())

print("\nOptimized structures saved to:", SDF_DIR)
print("Optimized coordinates saved to:", COORD_DIR)

print("\nOptimization index saved successfully.")

if failed:
    print("\nFailed drugs:")
    print(failed)

if not_converged:
    print("\nDrugs requiring further optimization:")
    print(not_converged)
