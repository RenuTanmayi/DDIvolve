
import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem

# ==========================================
# PATHS
# ==========================================

INPUT_FILE = "data/processed/clean_drugs.csv"

OUTPUT_DIR = "data/processed/molecular_features/3d_conformers"

SDF_DIR = os.path.join(OUTPUT_DIR, "structures")
COORD_DIR = os.path.join(OUTPUT_DIR, "coordinates")

os.makedirs(SDF_DIR, exist_ok=True)
os.makedirs(COORD_DIR, exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(INPUT_FILE)

successful = 0
failed = []

conformer_index = []

print("Generating 3D molecular conformers...\n")

# ==========================================
# GENERATE 3D CONFORMERS
# ==========================================

for idx, row in df.iterrows():

    drug_name = row["drug_name"]
    smiles = row["canonical_smiles"]

    if pd.isna(smiles):
        failed.append(drug_name)
        continue

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None or mol.GetNumAtoms() == 0:
        failed.append(drug_name)
        continue

    try:

        # Add explicit hydrogen atoms
        mol = Chem.AddHs(mol)

        # Configure ETKDGv3
        params = AllChem.ETKDGv3()

        params.randomSeed = 42
        params.useRandomCoords = False

        # Generate 3D conformer
        result = AllChem.EmbedMolecule(
            mol,
            params
        )

        if result != 0:
            failed.append(drug_name)
            print(f"Embedding failed: {drug_name}")
            continue

        # Extract 3D coordinates
        conformer = mol.GetConformer()

        coordinates = np.array(
            conformer.GetPositions(),
            dtype=np.float32
        )

        # Generate filenames
        sdf_file = f"conformer_{idx:04d}.sdf"
        coord_file = f"coordinates_{idx:04d}.npz"

        sdf_path = os.path.join(SDF_DIR, sdf_file)
        coord_path = os.path.join(COORD_DIR, coord_file)

        # Save 3D molecular structure
        Chem.MolToMolFile(
            mol,
            sdf_path
        )

        # Save 3D coordinates
        np.savez_compressed(
            coord_path,
            drug_name=np.array(drug_name),
            smiles=np.array(str(smiles)),
            coordinates=coordinates,
            atom_symbols=np.array([
                atom.GetSymbol()
                for atom in mol.GetAtoms()
            ])
        )

        conformer_index.append({

            "drug_name": drug_name,
            "smiles": str(smiles),
            "sdf_file": sdf_file,
            "coordinate_file": coord_file,
            "num_atoms_with_hydrogens": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms()

        })

        successful += 1

    except Exception as e:

        failed.append(drug_name)
        print(f"Error processing {drug_name}: {e}")

# ==========================================
# SAVE INDEX
# ==========================================

index_df = pd.DataFrame(conformer_index)

index_df.to_csv(
    os.path.join(OUTPUT_DIR, "3d_conformer_index.csv"),
    index=False
)

# ==========================================
# RESULTS
# ==========================================

print("\n========== 3D CONFORMER RESULTS ==========")

print("Total drugs:", len(df))
print("Successfully generated:", successful)
print("Failed:", len(failed))

if successful > 0:

    sample = np.load(
        os.path.join(
            COORD_DIR,
            conformer_index[0]["coordinate_file"]
        )
    )

    print(
        "Sample coordinate dimensions:",
        sample["coordinates"].shape
    )

print("\nSDF structures saved to:", SDF_DIR)
print("Coordinates saved to:", COORD_DIR)

print("\nIndex saved successfully.")

if failed:
    print("\nDrugs requiring review:")
    print(failed)
