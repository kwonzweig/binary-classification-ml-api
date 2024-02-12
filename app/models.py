from typing import List, Union

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """
    Represents a single data point with features for prediction.
    """

    features: List[Union[float, int, str] | None] = Field(
        ..., example=[5.1, None, 1.4, 0.2]
    )


class PredictionRequest(BaseModel):
    """
    Request model for making predictions.
    It contains a list of data points.
    """

    data: List[DataPoint]


class PredictionResult(BaseModel):
    """
    Represents the prediction result for a single data point.
    """

    label: int = Field(..., description="The predicted label (0 or 1).")
    probability: float = Field(..., description="The probability of the prediction.")


class PredictionResponse(BaseModel):
    """
    Response model for predictions.
    It contains a list of prediction results.
    """

    predictions: List[PredictionResult]


class TrainingResponse(BaseModel):
    """
    Response model for training completion.
    """

    message: str = Field(..., example="Model trained successfully.")
    accuracy: float = Field(..., description="The accuracy of the trained model.")


class EvaluationMetrics(BaseModel):
    """
    Schema for evaluation metrics of the model.
    """

    accuracy: float
    roc_auc: float
    pr_auc: float
    log_loss: float
    brier_score: float


class TrainingEvaluationResponse(BaseModel):
    """
    Response model for training completion with evaluation metrics.
    """

    training_scores: EvaluationMetrics
    testing_scores: EvaluationMetrics
