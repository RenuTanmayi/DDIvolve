
import numpy as np
import xgboost as xgb

from src.inference.predict import build_pair_features


DESCRIPTOR_NAMES = [
    "Molecular weight",
    "Exact molecular weight",
    "LogP",
    "Hydrogen bond donors",
    "Hydrogen bond acceptors",
    "TPSA",
    "Rotatable bonds",
    "QED",
    "Atom count",
    "Bond count",
    "Ring count",
    "Aromatic ring count",
    "Aliphatic ring count",
    "Fraction CSP3",
    "Heavy atom count",
    "Formal charge",
]


def get_feature_names():
    """Return feature names in the exact training-input order."""

    names = []

    # 64 descriptor-derived features
    for drug in ["Drug 1", "Drug 2"]:
        for descriptor in DESCRIPTOR_NAMES:
            names.append(f"{drug} - {descriptor}")

    for descriptor in DESCRIPTOR_NAMES:
        names.append(f"Absolute difference - {descriptor}")

    for descriptor in DESCRIPTOR_NAMES:
        names.append(f"Interaction product - {descriptor}")

    # 8192 fingerprint-derived features
    for group in [
        "Drug 1 Morgan fingerprint",
        "Drug 2 Morgan fingerprint",
        "Morgan fingerprint XOR",
        "Morgan fingerprint AND",
    ]:
        for bit in range(2048):
            names.append(f"{group} - Bit {bit}")

    if len(names) != 8256:
        raise ValueError(f"Expected 8256 feature names, got {len(names)}")

    return names


def get_feature_group(name):
    if name.startswith("Drug 1 -"):
        return "Drug 1 descriptors"
    if name.startswith("Drug 2 -"):
        return "Drug 2 descriptors"
    if name.startswith("Absolute difference"):
        return "Descriptor differences"
    if name.startswith("Interaction product"):
        return "Descriptor interactions"
    if name.startswith("Drug 1 Morgan"):
        return "Drug 1 fingerprint"
    if name.startswith("Drug 2 Morgan"):
        return "Drug 2 fingerprint"
    if name.startswith("Morgan fingerprint XOR"):
        return "Fingerprint XOR"
    if name.startswith("Morgan fingerprint AND"):
        return "Fingerprint AND"
    return "Other"


def explain_pair(predictor, smiles1, smiles2, top_k=20):
    """
    Explain the predicted interaction family using XGBoost TreeSHAP.

    SHAP contributions are in raw-margin units, not probabilities.
    """

    features = build_pair_features(smiles1, smiles2)
    dmatrix = xgb.DMatrix(features)

    # Obtain the model's original prediction probabilities.
    probabilities = np.asarray(
        predictor.model.predict(dmatrix),
        dtype=np.float64,
    ).reshape(-1)

    predicted_id = int(np.argmax(probabilities))
    predicted_family = predictor.classes[predicted_id]

    # XGBoost's native TreeSHAP contributions.
    contributions = np.asarray(
        predictor.model.predict(
            dmatrix,
            pred_contribs=True,
            approx_contribs=False,
        ),
        dtype=np.float64,
    )

    n_classes = len(predictor.classes)

    if contributions.ndim == 3:
        contributions = contributions[0]

    elif contributions.ndim == 2:
        if contributions.shape == (n_classes, 8257):
            pass
        elif contributions.shape == (1, n_classes * 8257):
            contributions = contributions.reshape(n_classes, 8257)
        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {contributions.shape}"
            )

    else:
        raise ValueError(
            f"Unexpected SHAP output dimensions: {contributions.shape}"
        )

    if contributions.shape != (n_classes, 8257):
        raise ValueError(
            f"Expected {(n_classes, 8257)}, "
            f"received {contributions.shape}"
        )

    names = get_feature_names()
    values = features.toarray().ravel()

    # Explain the predicted class.
    class_contributions = contributions[predicted_id]

    feature_shap = class_contributions[:-1]
    base_value = float(class_contributions[-1])

    order = np.argsort(np.abs(feature_shap))[::-1][:top_k]

    top_features = []

    for index in order:
        shap_value = float(feature_shap[index])

        top_features.append({
            "feature": names[index],
            "feature_index": int(index),
            "feature_value": float(values[index]),
            "shap_value": shap_value,
            "group": get_feature_group(names[index]),
            "direction": (
                "Supports predicted class"
                if shap_value > 0
                else "Opposes predicted class"
                if shap_value < 0
                else "No contribution"
            ),
        })

    # Aggregate absolute SHAP importance by feature group.
    group_importance = {}

    for index, shap_value in enumerate(feature_shap):
        group = get_feature_group(names[index])
        group_importance[group] = (
            group_importance.get(group, 0.0)
            + abs(float(shap_value))
        )

    group_summary = [
        {
            "group": group,
            "importance": float(importance),
        }
        for group, importance in sorted(
            group_importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    return {
        "predicted_family": predicted_family,
        "class_id": predicted_id,
        "confidence": float(probabilities[predicted_id]),
        "probabilities": {
            name: float(probabilities[i])
            for i, name in enumerate(predictor.classes)
        },
        "base_value": base_value,
        "top_features": top_features,
        "group_importance": group_summary,
        "explanation_type": "XGBoost TreeSHAP",
        "interpretation_note": (
            "SHAP values explain contributions to the model's raw "
            "class score. They do not establish biological causality "
            "or clinical severity."
        ),
    }
