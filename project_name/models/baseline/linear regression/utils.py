import sys
sys.path.append("../")
import pickle
from data.path_grapper import get_train_data_folders
from model import LinearModelHandler  # Import the class instead of functions
from data_loader import LinearRegressionDataset



def load_linear_data(split: str):
    dataset = LinearRegressionDataset(split)
    X, y = dataset.get_all()
    return X, y


def main():
    # Load train/test data similarly to CNNDataset usage
    X_train, y_train = load_linear_data("train")
    X_test, y_test = load_linear_data("test")

    # Now use your existing linear model training and evaluation
    model_handler = LinearModelHandler()
    model_handler.train(X_train, y_train)

    rmse, mae, inference_time = model_handler.evaluate(X_test, y_test)

    print(f"RMSE: {rmse}, MAE: {mae}, Inference Time: {inference_time:.4f} s")


if __name__ == "__main__":
    main()
