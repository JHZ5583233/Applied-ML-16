from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple
import time
import numpy as np

class LinearModelHandler:
    """
    Handles training, prediction, and evaluation of a linear regression model.
    """

    def __init__(self):
        self.model: LinearRegression = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the linear regression model."""
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        end_time = time.time()

        rmse_score = mean_squared_error(y_test, y_pred, squared=False)
        mae_score = mean_absolute_error(y_test, y_pred)
        inference_time = end_time - start_time

        print(f"RMSE: {rmse_score}")
        print(f"MAE: {mae_score}")

        return rmse_score, mae_score, inference_time
