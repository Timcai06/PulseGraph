from __future__ import annotations

from typing import Any


def classification_output(
    *,
    label: int,
    prediction: int,
    probabilities: list[float],
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    if 0 <= prediction < len(probabilities):
        confidence = probabilities[prediction]
    else:
        confidence = max(probabilities) if probabilities else 0.0
    return {
        "kind": "classification",
        "label": label,
        "prediction": prediction,
        "confidence": float(confidence),
        "probabilities": probabilities,
        "class_names": class_names,
    }
