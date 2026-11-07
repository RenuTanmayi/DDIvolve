
import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# File paths
INPUT_FILE = "data/processed/clean_drugs.csv"
OUTPUT_DIR = "data/processed/molecular_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load drug dataset
df = pd.read_csv(INPUT_FILE)

# Initialize Morgan fingerprint generator
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

fingerprints = []
drug_names = []
failed = []

print("Generating Morgan fingerprints...\n")

for _, row in df.iterrows():

    drug = row["drug_name"]
    smiles = row["canonical_smiles"]

    if pd.isna(smiles):
        failed.append(drug)
        continue

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        failed.append(drug)
        continue

    fp = generator.GetFingerprint(mol)

    # Convert fingerprint into a 2048-bit array
    arr = np.zeros((2048,), dtype=np.uint8)
    for bit in fp.GetOnBits():
        arr[bit] = 1

    fingerprints.append(arr)
    drug_names.append(drug)

# Convert to NumPy array
fingerprints = np.array(fingerprints, dtype=np.uint8)

# Save fingerprints and corresponding drug names
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "morgan_fingerprints.npz"),
    fingerprints=fingerprints,
    drug_names=np.array(drug_names)
)

print("========== MORGAN FINGERPRINT RESULTS ==========")
print("Total drugs:", len(df))
print("Successfully generated:", len(fingerprints))
print("Failed:", len(failed))
print("Fingerprint size:", fingerprints.shape)

print("\nSaved to:")
print(os.path.join(OUTPUT_DIR, "morgan_fingerprints.npz"))

if failed:
    print("\nDrugs requiring review:")
    print(failed)
