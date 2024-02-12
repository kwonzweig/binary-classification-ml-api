import pytest
import requests

BASE_URL = "http://localhost:8000"  # Update if your application is hosted differently


def test_upload():
    """
    Test the /upload/ endpoint for uploading a dataset.
    """
    files = {'file': open('unit_test_dataset.csv', 'rb')}
    response = requests.post(f"{BASE_URL}/upload/", files=files)
    assert response.status_code == 200
    assert "Successfully uploaded" in response.json().get("message")


def test_train():
    """
    Test the /train/ endpoint to ensure the model trains and returns evaluation metrics.
    """
    files = {'file': open('unit_test_dataset.csv', 'rb')}
    response = requests.post(f"{BASE_URL}/train/", files=files)
    assert response.status_code == 200
    # Check for the presence of evaluation metrics in the response
    metrics = response.json()
    assert "training_scores" in metrics
    assert "testing_scores" in metrics
    # Example: assert a specific metric is returned; adjust based on your evaluation metrics
    assert "accuracy" in metrics["training_scores"]
    assert metrics["training_scores"]["accuracy"] > 0


def test_predict():
    """
    Test the /predict/ endpoint to ensure it makes predictions.
    """
    # Adjust this payload to match the expected input format of your model
    payload = {
        'data': [
            {
                'features': [30, 'management', 'single', 'tertiary', 'no', 3773, 'yes', 'no', None, 27, 'may',
                             99, 1, -1, 0, None]},
            {
                'features': [39, 'technician', 'single', None, 'no', 45248, 'yes', 'no', None, 6, 'may',
                             1623, 1, -1, 0, None]},
            {
                'features': [51, 'admin.', 'single', 'secondary', 'no', 895, 'no', 'no', 'cellular', 23,
                             'jul', 638, 2, -1, 0, None]},
            {
                'features': [26, 'self-employed', 'single', 'tertiary', 'no', 82, 'yes', 'no', 'cellular',
                             17, 'jul', 200, 1, -1, 0, None]},
            {
                'features': [21, 'student', 'single', 'secondary', 'no', 2488, 'no', 'no', 'cellular', 12,
                             'oct', 180, 1, -1, 0, None]}
        ]
    }

    response = requests.post(f"{BASE_URL}/predict/", json=payload)
    assert response.status_code == 200
    predictions = response.json().get("predictions")
    assert len(predictions) > 0
    # Example: Check the structure of the prediction response
    assert "label" in predictions[0]
    assert "probability" in predictions[0]


if __name__ == "__main__":
    pytest.main()
