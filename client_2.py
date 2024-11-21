import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import tasks as ts
from typing import List, Dict, Tuple
from flwr.common import NDArrays


glob_round = 0

class CustomClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def get_parameters(self, config: Dict[str, int]) -> NDArrays:
        return ts.get_model_parameters(self.model)
    
    def fit(self, parameters: NDArrays, config: Dict[str, int]) -> Tuple[NDArrays, int, Dict]:
        global glob_round
        glob_round += 1

        ts.set_model_parameters(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)

        return ts.get_model_parameters(self.model), len(self.X_train), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, int]) -> Tuple[float, int, Dict]:
        global glob_round
        ts.set_model_parameters(self.model, parameters)

        if glob_round < 50:
            y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
            y_pred = self.model.predict(self.X_val)

            loss = log_loss(self.y_val, y_pred_proba)
            accuracy = accuracy_score(self.y_val, y_pred)
            roc_auc = roc_auc_score(self.y_val, y_pred_proba)
            f1 = f1_score(self.y_val, y_pred)

            print(f"Validation Metrics (Round {glob_round}):")
            print(f" - Loss: {loss}")
            print(f" - Accuracy: {accuracy}")
            print(f" - ROC AUC: {roc_auc}")
            print(f" - F1 Score: {f1}")
        else:
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            y_pred = self.model.predict(self.X_test)

            loss = log_loss(self.y_test, y_pred_proba)
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            f1 = f1_score(self.y_test, y_pred)

            print(f"Test Metrics (Final Round):")
            print(f" - Loss: {loss}")
            print(f" - Accuracy: {accuracy}")
            print(f" - ROC AUC: {roc_auc}")
            print(f" - F1 Score: {f1}")

        return loss, len(self.X_val), {"accuracy": accuracy, "roc_auc": roc_auc, "f1-score": f1}

if __name__ == "__main__":
    
    path_train = "./NON_IID_2.csv"
    path_test = "./datas/test_glob.csv"

    data_train = pd.read_csv(path_train)
    data_test = pd.read_csv(path_test)
    X_test, y_test = data_test.drop(columns="Fraud"), data_test.Fraud

    X_train, X_val, X_test, y_train, y_val  =  ts.prepare_data(data_train, X_test)

    model = LogisticRegression(max_iter=5, warm_start=True)

    ts.set_initial_parameters(model)

    client = CustomClient(model, X_train, X_val, X_test, y_train, y_val, y_test)

    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
