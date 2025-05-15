from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import time
from typing import List, Tuple, Optional, Any

def train_linear_models(
    data: List[List[Any]], num_of_models: int
) -> List[LinearRegression]:
    """
    Predict the values of the input data using the model.

    Args:
        data (List[List[Any]]): A list containing the input data.
        num_of_models (int): Number of models to train.

    Returns:
        List[LinearRegression]: A list containing the trained linear regression models.
    """
    trained_models = []
    X_train = data[0]

    for i in range(num_of_models):
        y_train = data[i+1]
        model = train_linear_regr(X_train, y_train)
        trained_models.append(model)

    return trained_models

def predict_linear_models(models: List[LinearRegression], X_test: List[Any]) -> List[Any]:
    """
    Predict the values of the input data using the linear regression model.

    Args:
        model (List[LinearRegression]): A list of linear regression models used for prediction.
        X_test (List[Any]): A list containing the input data.

    Returns:
        List[Any]: A list containing the predicted values.
    """
    y_preds = []
    for model in models:
        y_pred = model.predict(X_test)
        y_preds.append(y_pred)
    return y_preds

def evaluate_linear_models(
    models: List[LinearRegression], X_test: List[Any], y_test: List[Any]
) -> Tuple[float, float, float]:
    """
    Predict the values of the input data using the model.

    Args:
        model (List[LinearRegression]): A list of linear regression models to be evaluated.
        X_test (List[Any]): A list containing the input data.
        y_test (List[Any]): A list containing the target data.

    Returns:
        Tuple[float, float, float]: A tuple containing the RMSE, MAE, and inference time.
    """
    rmse, mae = [], []
    start_time = time.time()
    for i, model in enumerate(models):
        y_pred = model.predict(X_test)
        rmse.append(root_mean_squared_error(y_test[i], y_pred))
        print("RMSE day " + str(i+1) + ": " + str(rmse[i]))
        mae.append(mean_absolute_error(y_test[i], y_pred))
        print("MAE day " + str(i+1) + ": " + str(mae[i]))

    end_time = time.time()
    inference_time = (end_time - start_time)

    return sum(rmse), sum(mae), inference_time

def train_linear_regr(
    X_train: List[Any], y_train: List[Any], model: Optional[LinearRegression] = None
) -> LinearRegression:
    """
    Train a (new) linear regression model.

    Args:
        X_train (List[Any]): A list containing the input data.
        y_train (List[Any]): A list containing the target data.
        model (Optional[LinearRegression]): An optional linear regression model to be trained.
                                            If no model is provided, a new model will be created.

    Returns:
        LinearRegression: A trained linear regression model.
    """
    if model is None:
        model = LinearRegression()
    model.fit(X_train, y_train)
    return model