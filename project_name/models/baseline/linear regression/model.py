from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from typing import List, Tuple, Any
import time

from .utils import train_linear_regr  # Assumes this is a helper function that returns a trained LinearRegression


class LinearModelHandler:
    """
    Handles training, prediction, and evaluation of a linear regression model.
    """

    def __init__(self):
        self.model: LinearRegression = None

    def train(self, X_train: List[Any], y_train: List[Any]) -> None:
        """
        Train the linear regression model.

        Args:
            X_train (List[Any]): Training features.
            y_train (List[Any]): Training targets.
        """
        self.model = train_linear_regr(X_train, y_train)

    def predict(self, X_test: List[Any]) -> List[Any]:
        """
        Predict using the trained model.

        Args:
            X_test (List[Any]): Test features.

        Returns:
            List[Any]: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X_test)

    def evaluate(self, X_test: List[Any], y_test: List[Any]) -> Tuple[float, float, float]:
        """
        Evaluate the trained model.

        Args:
            X_test (List[Any]): Test features.
            y_test (List[Any]): True target values.

        Returns:
            Tuple[float, float, float]: RMSE, MAE, and inference time.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        end_time = time.time()

        rmse_score = root_mean_squared_error(y_test, y_pred)
        mae_score = mean_absolute_error(y_test, y_pred)
        inference_time = end_time - start_time

        print(f"RMSE: {rmse_score}")
        print(f"MAE: {mae_score}")

        return rmse_score, mae_score, inference_time
