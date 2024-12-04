import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
import pandas as pd
from fed_env.fed_tasks_sklearn import (
                            get_data,
                            prepare_data,
                            set_initial_parameters,
                            get_model_parameters,
                            set_model_parameters,
                            save_metrics_json

)
from typing import Dict, Tuple
from flwr.common import NDArrays
import logging
from datetime import datetime
import os 



log_dir = "./fed_env/process"
os.makedirs(log_dir, exist_ok=True)


log_filename = os.path.join(log_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)





glob_round = 0

class CustomClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        self.losses = []
        self.ROC_AUCs = []
        self.ACCURACYs = []
        self.F1s = []

    def get_parameters(self, config: Dict[str, int]) -> NDArrays:
        return get_model_parameters(self.model)
    
    def fit(self, parameters: NDArrays, config: Dict[str, int]) -> Tuple[NDArrays, int, Dict]:
        global glob_round
        glob_round += 1

        set_model_parameters(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)

        return get_model_parameters(self.model), len(self.X_train), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, int]) -> Tuple[float, int, Dict]:
        global glob_round
        set_model_parameters(self.model, parameters)
        try:
            if glob_round < 50:
                X_eval = self.X_val
                y_eval = self.y_val
            else:
                X_eval = self.X_test
                y_eval = self.y_test

            y_pred_proba = self.model.predict_proba(X_eval)[:, 1]
            y_pred = self.model.predict(X_eval)

            loss = log_loss(y_eval, y_pred_proba)
            accuracy = accuracy_score(y_eval, y_pred)
            roc_auc = roc_auc_score(y_eval, y_pred_proba)
            f1 = f1_score(y_eval, y_pred)
            self.losses.append(loss)
            self.ROC_AUCs.append(roc_auc)
            self.ACCURACYs.append(accuracy)
            self.F1s.append(f1)

            
            logging.info(f"Evaluation Metrics (Round {glob_round}):")
            logging.info(f" - Loss: {loss}")
            logging.info(f" - Accuracy: {accuracy}")
            logging.info(f" - ROC AUC: {roc_auc}")
            logging.info(f" - F1 Score: {f1}")

            return loss, len(X_eval), {"accuracy": accuracy, "roc_auc": roc_auc, "f1-score": f1}
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            return float('inf'), 0, {}

if __name__ == "__main__":
    TRAIN_DATA_PATH = "./datas/NON_IID_FL_1.csv"
    TEST_SAMPLE_PATH = "./datas/TEST_SAMPLE.csv"
    SAVE_PATH = "./fed_env/results/noniid.json"

    data_noniid = get_data(TRAIN_DATA_PATH)
    data_test = pd.read_csv(TEST_SAMPLE_PATH)

 
    X_train, X_val, X_test, y_train, y_val = prepare_data(data_noniid, data_test.drop(columns="Fraud"))
    y_test = data_test.Fraud.values

    model = LogisticRegression(max_iter=10, warm_start=True)
    set_initial_parameters(model)

    client = CustomClient(model, X_train, X_val, X_test, y_train,y_val, y_test)

    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
    save_metrics_json(client, "fed_avg_noniid", SAVE_PATH)

