from model import LinearModelHandler
from data_loader import LinearRegressionDataset
import numpy as np


class LinearRegressionPipeline:
    """
    Pipeline to handle data loading, training, 
    and evaluation of a linear regression model.
    """

    def __init__(self):
        """Initializes the linear reg model"""
        self.model_handler = LinearModelHandler()
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_test: np.ndarray = None

    def load_data(self) -> None:
        """Loads the training and test data."""
        print("Loading training data...")
        train_dataset = LinearRegressionDataset("train")
        self.X_train, self.y_train = train_dataset.get_all()

        print("Loading test data...")
        test_dataset = LinearRegressionDataset("test")
        self.X_test, self.y_test = test_dataset.get_all()

    def train_model(self) -> None:
        """Trains the linear regression model using loaded training data."""
        print("Training model...")
        self.model_handler.train(self.X_train, self.y_train)

    def evaluate_model(self) -> None:
        """Evaluates the trained model on the test data."""
        print("Evaluating model...")
        rmse, mae, inference_time = self.model_handler.evaluate(
            self.X_test, self.y_test)
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Inference Time: {inference_time:.4f} s")

    def run(self) -> None:
        """Executes the full pipeline."""
        self.load_data()
        self.train_model()
        self.evaluate_model()


def main():
    """
    Main function for running the pipeline.
    """
    pipeline = LinearRegressionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
