import pandas as pd
from joblib import dump, load
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, accuracy_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


class BinaryClassifierModel:
    def __init__(self, input_features, numerical_features, categorical_features):
        """
        Initialize the binary classification model along with its preprocessing pipeline.
        """
        self.input_features = input_features

        # Define preprocessing for numerical columns
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Define preprocessing for categorical columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Create the preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create the full pipeline
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the pipeline (preprocessing + XGBoost Classifier) model and evaluate it.
        """

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the pipeline
        self.pipeline.fit(X_train, y_train)

        # Predictions for evaluation
        y_train_pred = self.pipeline.predict(X_train)
        y_train_proba = self.pipeline.predict_proba(X_train)[:, 1]
        y_test_pred = self.pipeline.predict(X_test)
        y_test_proba = self.pipeline.predict_proba(X_test)[:, 1]

        # Calculate scores
        train_scores = self._calculate_scores(y_train, y_train_pred, y_train_proba)
        test_scores = self._calculate_scores(y_test, y_test_pred, y_test_proba)

        return {'training_scores': train_scores, 'testing_scores': test_scores}

    def predict(self, X: pd.DataFrame):
        """
        Make predictions with the trained pipeline.
        """
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)[:, 1]  # Probability of positive class
        return predictions, probabilities

    def _calculate_scores(self, y_true, y_pred, y_proba):
        """
        Helper method to calculate evaluation metrics.
        """
        scores = {
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'pr_auc': average_precision_score(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba)
        }
        return scores


# Adjusting the external interaction utility functions
def train_model(df: pd.DataFrame):
    """
    Train the binary classification model and return evaluation metrics.
    """

    # Split data into features and target
    X = df.iloc[:, :-1]  # All rows, all columns except the last
    y = df.iloc[:, -1]  # All rows, only the last column

    # Identify numerical and categorical features
    numerical_features, categorical_features = identify_feature_types(X)

    # Train the model
    model = BinaryClassifierModel(X.columns, numerical_features, categorical_features)
    evaluation_summary = model.train_and_evaluate(X, y)

    # Save the trained pipeline to a file
    dump(model, 'trained_model.joblib')

    return evaluation_summary


def predict_model(X: pd.DataFrame):
    """
    Predict the labels and probabilities for given data.
    """
    try:
        # Attempt to load the trained model class
        model = load('trained_model.joblib')
        pipeline = model.pipeline
    except FileNotFoundError:
        raise Exception("Model not found. Please train the model before prediction.")

    # Ensure the input DataFrame columns match the expected feature names
    # This step assumes X is already prepared with correct order and number of features
    # If X comes with its own column names, you might want to align or reorder them as per feature_names
    # For demonstration, we're directly assigning the expected feature names to X
    X.columns = model.input_features

    # Make predictions
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]  # Probability of positive class

    return predictions, probabilities


def identify_feature_types(df: pd.DataFrame):
    """
    Identify numerical and categorical features in the DataFrame.

    Parameters:
    - df: pd.DataFrame - The input DataFrame.

    Returns:
    - numerical_features: List[str] - A list of names of numerical features.
    - categorical_features: List[str] - A list of names of categorical features.
    """
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical_features, categorical_features
