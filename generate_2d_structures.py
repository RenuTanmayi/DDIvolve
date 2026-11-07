
import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

# ==========================================
# PATHS
# ==========================================

INPUT_FILE = "data/processed/clean_drugs.csv"

OUTPUT_DIR = "data/processed/molecular_features/2d_structures"

IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
COORD_DIR = os.path.join(OUTPUT_DIR, "coordinates")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(COORD_DIR, exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(INPUT_FILE)

successful = 0
failed = []

structure_index = []

print("Generating 2D molecular structures...\n")

# ==========================================
# GENERATE 2D STRUCTURES
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

        # Generate 2D molecular coordinates
        rdDepictor.Compute2DCoords(mol)

        # Generate filenames
        image_file = f"structure_{idx:04d}.png"
        coord_file = f"coordinates_{idx:04d}.npz"

        image_path = os.path.join(IMAGE_DIR, image_file)
        coord_path = os.path.join(COORD_DIR, coord_file)

        # Save 2D molecular structure image
        Draw.MolToFile(
            mol,
            image_path,
            size=(500, 400),
            kekulize=True,
            wedgeBonds=True,
            imageType="png"
        )

        # Extract 2D atom coordinates
        conformer = mol.GetConformer()

        coordinates = np.array(
            conformer.GetPositions(),
            dtype=np.float32
        )

        # Save coordinates
        np.savez_compressed(
            coord_path,
            drug_name=np.array(drug_name),
            smiles=np.array(str(smiles)),
            atom_coordinates=coordinates
        )

        structure_index.append({
            "drug_name": drug_name,
            "smiles": str(smiles),
            "image_file": image_file,
            "coordinate_file": coord_file,
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds()
        })

        successful += 1

    except Exception as e:

        failed.append(drug_name)
        print(f"Error processing {drug_name}: {e}")

# ==========================================
# SAVE INDEX
# ==========================================

index_df = pd.DataFrame(structure_index)

index_df.to_csv(
    os.path.join(OUTPUT_DIR, "2d_structure_index.csv"),
    index=False
)

# ==========================================
# RESULTS
# ==========================================

print("\n========== 2D STRUCTURE RESULTS ==========")

print("Total drugs:", len(df))
print("Successfully generated:", successful)
print("Failed:", len(failed))

print("\nImage dimensions: 500 x 400 pixels")
print("Coordinate dimensions: (number_of_atoms, 3)")

print("\nImages saved to:", IMAGE_DIR)
print("Coordinates saved to:", COORD_DIR)

print("\nIndex saved successfully.")

if failed:
    print("\nDrugs requiring review:")
    print(failed)
