
from pathlib import Path
import json

import joblib
import numpy as np
import scipy.sparse as sp
import xgboost as xgb

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors, QED


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "severity_module" / "models" / "stage2_xgboost"

MODEL_PATH = MODEL_DIR / "ddv_interaction_xgboost.json"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

FP_RADIUS = 2
FP_BITS = 2048
EXPECTED_FEATURES = 8256
EXPECTED_CLASSES = 9


def extract_molecule_features(smiles):
    """Extract 16 molecular descriptors and a 2048-bit Morgan fingerprint."""

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        round(Descriptors.MolLogP(mol), 3),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumRotatableBonds(mol),
        round(QED.qed(mol), 3),
        mol.GetNumAtoms(),
        mol.GetNumBonds(),
        mol.GetRingInfo().NumRings(),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
        Chem.GetFormalCharge(mol),
    ]

    generator = AllChem.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_BITS,
    )

    fingerprint = generator.GetFingerprint(mol)

    fp_array = np.zeros(FP_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, fp_array)

    return np.asarray(descriptors, dtype=np.float32), fp_array


def build_pair_features(smiles1, smiles2):
    """Construct the original 8256-feature model input."""

    desc1, fp1 = extract_molecule_features(smiles1)
    desc2, fp2 = extract_molecule_features(smiles2)

    descriptor_features = np.concatenate([
        desc1,
        desc2,
        np.abs(desc1 - desc2),
        desc1 * desc2,
    ]).astype(np.float32)

    fingerprint_features = np.concatenate([
        fp1,
        fp2,
        np.bitwise_xor(fp1, fp2),
        np.bitwise_and(fp1, fp2),
    ]).astype(np.uint8)

    descriptor_features = np.nan_to_num(
        descriptor_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    combined = np.concatenate([
        descriptor_features,
        fingerprint_features.astype(np.float32),
    ])

    if len(combined) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, got {len(combined)}"
        )

    return sp.csr_matrix(
        combined.reshape(1, -1),
        dtype=np.float32,
    )


class DDIvolvePredictor:

    def __init__(self):

        for path in [MODEL_PATH, ENCODER_PATH, METADATA_PATH]:
            if not path.exists():
                raise FileNotFoundError(f"Missing model file: {path}")

        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_PATH))

        self.encoder = joblib.load(ENCODER_PATH)

        with open(METADATA_PATH, encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.classes = self.metadata["classes"]

        if len(self.classes) != EXPECTED_CLASSES:
            raise ValueError("Unexpected interaction-family class count.")

        if self.model.num_features() != EXPECTED_FEATURES:
            raise ValueError(
                f"Model feature count is not {EXPECTED_FEATURES}."
            )

        encoder_classes = [str(c) for c in self.encoder.classes_]
        metadata_classes = [str(c) for c in self.classes]

        if encoder_classes != metadata_classes:
            raise ValueError(
                "Label encoder and model metadata class ordering differ."
            )

    def predict(self, smiles1, smiles2):

        features = build_pair_features(smiles1, smiles2)

        dmatrix = xgb.DMatrix(features)

        probabilities = np.asarray(
            self.model.predict(dmatrix),
            dtype=np.float64,
        ).reshape(-1)

        if len(probabilities) != len(self.classes):
            raise ValueError("Unexpected prediction probability count.")

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Model returned non-finite probabilities.")

        if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
            raise ValueError("Prediction probabilities do not sum to 1.")

        predicted_id = int(np.argmax(probabilities))

        return {
            "drug1_smiles": smiles1,
            "drug2_smiles": smiles2,
            "interaction_family": self.classes[predicted_id],
            "class_id": predicted_id,
            "confidence": float(probabilities[predicted_id]),
            "probabilities": {
                name: float(probabilities[i])
                for i, name in enumerate(self.classes)
            },
        }
