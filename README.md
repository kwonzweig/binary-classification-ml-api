
# Binary Classification API

## Introduction
This project provides a simple API for binary classification tasks using FastAPI. It's designed to help beginners learn how to build and deploy machine learning APIs. The API allows users to upload datasets, train a model, and make predictions.

## Setup and Installation

### Requirements
- Python 3.8+
- Docker (optional for containerization)

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/kwonzweig/binary-classification-api.git
   cd binary-classification-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be available at `http://localhost:8000`.

### Using Docker
1. Build the Docker image:
   ```bash
   docker build -t binary-classification-api .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:80 binary-classification-api
   ```
   The API will be accessible at `http://localhost:8000`.

## API Usage

### Train Model
- **Endpoint**: `POST /train/`
- **Description**: Train the model using the uploaded dataset.
- **cURL Example**:
  ```bash
  curl -X 'POST' 'http://localhost:8000/train/' -F 'file=@path_to_your_dataset.csv;type=text/csv'
  ```

### Make Prediction
- **Endpoint**: `POST /predict/`
- **Description**: Make predictions with the trained model.
- **cURL Example**:
  ```bash
  curl -X 'POST' 'http://localhost:8000/predict/' -H 'Content-Type: application/json' -d '{"data": [{"features": [30, "management", "single", "tertiary", "no", 3773, "yes", "no", null, 27, "may", 99, 1, -1, 0, null]}, ...]}'
  ```

## Testing the API with `demo_api_calls.py`

### About the Bank Marketing Dataset

The Bank Marketing dataset is utilized in this project to demonstrate the API's capabilities. This dataset originates from direct marketing campaigns, specifically phone calls made by a Portuguese banking institution. The primary classification objective is to predict whether a client will subscribe to a term deposit, denoted by the variable `y`. The dataset provides a realistic context for testing machine learning models, showcasing how data from direct marketing efforts can be leveraged to inform business decisions.

### Prerequisites

Before running the script, ensure you have the following prerequisites:

1. Python 3.x installed on your system.
2. Access to a terminal or command prompt.
3. The Bank Marketing dataset downloaded and prepared according to the script's expectations. You can find details about the dataset used in this script from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).

### Running the Script

   ```
   python demo_api_calls.py
   ```


After running the script successfully, you will see the following output indicating each step's result:

- **Train Response**:
  ```python
  {'training_scores': {'accuracy': 0.9630896837739762, 'roc_auc': 0.9855929562230337, 'pr_auc': 0.9220665310189687, 'log_loss': 0.11033718942637911, 'brier_score': 0.030947935079333938}, 'testing_scores': {'accuracy': 0.904202377661045, 'roc_auc': 0.9278738510885942, 'pr_auc': 0.601454074096591, 'log_loss': 0.20962202748945483, 'brier_score': 0.06515554924358767}}
  ```

- **Predict Request**:
  ```python
  {'data': [{'features': [30, 'management', 'single', 'tertiary', 'no', 3773, 'yes', 'no', None, 27, 'may', 99, 1, -1, 0, None]}, {'features': [39, 'technician', 'single', None, 'no', 45248, 'yes', 'no', None, 6, 'may', 1623, 1, -1, 0, None]}, {'features': [51, 'admin.', 'single', 'secondary', 'no', 895, 'no', 'no', 'cellular', 23, 'jul', 638, 2, -1, 0, None]}, {'features': [26, 'self-employed', 'single', 'tertiary', 'no', 82, 'yes', 'no', 'cellular', 17, 'jul', 200, 1, -1, 0, None]}, {'features': [21, 'student', 'single', 'secondary', 'no', 2488, 'no', 'no', 'cellular', 12, 'oct', 180, 1, -1, 0, None]}]}
  ```

- **Predict Response**:
  ```python
  {'predictions': [{'label': 0, 'probability': 0.004984984174370766}, {'label': 1, 'probability': 0.6178786158561707}, {'label': 0, 'probability': 0.26504549384117126}, {'label': 0, 'probability': 0.010345274582505226}, {'label': 1, 'probability': 0.6866607666015625}]}
  ```

## Development

This project is designed for educational purposes to demonstrate how to build a simple ML API with FastAPI. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
