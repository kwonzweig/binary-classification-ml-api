import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException

from app.ml_model.model import train_model, predict_model
from app.models import PredictionRequest, PredictionResponse, PredictionResult, TrainingEvaluationResponse

app = FastAPI(title="Binary Classification API", description="A simple API for binary classification tasks",
              version="1.0")


@app.post("/train/", response_model=TrainingEvaluationResponse)
async def train(file: UploadFile = File(...)):
    """
    Train the model on the uploaded dataset and return evaluation metrics.
    Assumes the last column is the target variable.
    """
    try:
        df = pd.read_csv(file.file)
        evaluation_summary = train_model(df)
        return evaluation_summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to train model: {e}")


@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions with the trained model. Expects a list of data points.
    """
    try:
        # Convert PredictionRequest to DataFrame
        data = pd.DataFrame([data_point.features for data_point in request.data])

        # Make predictions
        predictions, probabilities = predict_model(data)
        results = [PredictionResult(label=label, probability=prob) for label, prob in zip(predictions, probabilities)]
        return PredictionResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to make predictions: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
