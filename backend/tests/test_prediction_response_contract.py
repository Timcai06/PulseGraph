from app.schemas import PredictionResponse


def test_classification_prediction_keeps_legacy_fields() -> None:
    response = PredictionResponse(
        task="classification",
        output={
            "kind": "classification",
            "label": 1,
            "prediction": 2,
            "probabilities": [0.1, 0.2, 0.7],
        },
        sample_index=0,
        label=1,
        prediction=2,
        weights="trained",
        sample_source="probe",
        image_shape=[1, 2, 2],
        image_pixels=[0.0, 0.1, 0.2, 0.3],
        probabilities=[0.1, 0.2, 0.7],
        graph={"nodes": [], "edges": []},
        layers=[],
    )

    assert response.task == "classification"
    assert response.label == 1
    assert response.prediction == 2
    assert response.probabilities == [0.1, 0.2, 0.7]


def test_detection_prediction_accepts_structured_output_without_classification_fields() -> None:
    response = PredictionResponse(
        task="detection",
        output={
            "kind": "detection",
            "boxes": [[0.1, 0.2, 0.8, 0.9]],
            "scores": [0.94],
            "labels": ["target"],
        },
        sample_index=0,
        weights="trained",
        sample_source="probe",
        image_shape=[3, 32, 32],
        image_pixels=[0.0] * (3 * 32 * 32),
        graph={"nodes": [], "edges": []},
        layers=[],
    )

    assert response.task == "detection"
    assert response.label is None
    assert response.prediction is None
    assert response.probabilities == []
    assert response.output["kind"] == "detection"
