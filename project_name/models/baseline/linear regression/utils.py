import sys
sys.path.append("../")
import pickle
from sklearn.linear_model import LinearRegression
from data.path_grapper import get_train_data_folders
from model import model


def main():
    train_data = get_train_data_folders("train")
    test_data = 0

    # Linear Regression
    model = model.train_linear_models(
        data=train_data, num_of_models=3
    )
    lr_rmse, lr_mae, lr_inference_time = model.evaluate_linear_models(
        model, test_data[0], [test_data[1], test_data[2], test_data[3]]
    )

    
    with open("linear_regression.pkl") as file:  
        pickle.dump(model, file)
    
    print("Linear Regression Metrics")
    print("\tRMSE: ", lr_rmse)
    print("\tMAE: ", lr_mae)
    print("\tInference time: ", lr_inference_time)

if __name__ == "__main__":
    main()
