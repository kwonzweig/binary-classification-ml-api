import os

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_train():
    """
    Test the /train/ endpoint to ensure the model trains and returns evaluation metrics.
    """
    # Load the demo training dataset
    with open(os.path.join("tests", "demo_train_dataset.csv"), "rb") as file:
        response = client.post("/train/", files={"file": file})
    assert response.status_code == 200
    metrics = response.json()
    assert "training_scores" in metrics
    assert "testing_scores" in metrics
    assert (
        metrics["training_scores"]["accuracy"] > 0.5
    )  # Example threshold, adjust based on expected performance


def test_predict():
    """
    Test the /predict/ endpoint to ensure it makes predictions with expected structure and values.
    """
    payload = {
        "data": [
            {
                "features": [
                    30,
                    "management",
                    "single",
                    "tertiary",
                    "no",
                    3773,
                    "yes",
                    "no",
                    None,
                    27,
                    "may",
                    99,
                    1,
                    -1,
                    0,
                    None,
                ]
            },
            {
                "features": [
                    39,
                    "technician",
                    "single",
                    None,
                    "no",
                    45248,
                    "yes",
                    "no",
                    None,
                    6,
                    "may",
                    1623,
                    1,
                    -1,
                    0,
                    None,
                ]
            },
            {
                "features": [
                    51,
                    "admin.",
                    "single",
                    "secondary",
                    "no",
                    895,
                    "no",
                    "no",
                    "cellular",
                    23,
                    "jul",
                    638,
                    2,
                    -1,
                    0,
                    None,
                ]
            },
            {
                "features": [
                    26,
                    "self-employed",
                    "single",
                    "tertiary",
                    "no",
                    82,
                    "yes",
                    "no",
                    "cellular",
                    17,
                    "jul",
                    200,
                    1,
                    -1,
                    0,
                    None,
                ]
            },
            {
                "features": [
                    21,
                    "student",
                    "single",
                    "secondary",
                    "no",
                    2488,
                    "no",
                    "no",
                    "cellular",
                    12,
                    "oct",
                    180,
                    1,
                    -1,
                    0,
                    None,
                ]
            },
        ]
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    predictions = response.json().get("predictions")
    assert len(predictions) == 5  # Ensuring we have a prediction for each input

    for prediction in predictions:
        assert "label" in prediction
        assert prediction["label"] in [0, 1]
        assert "probability" in prediction
        assert 0 <= prediction["probability"] <= 1
