import os
import json
import pickle
import warnings
import argparse
import flwr as fl
import numpy as np
import tasks as ts
import pandas as pd

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from typing import Dict





class CustomClient(fl.client.NumPyClient):

    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.losses = []
        self.ROC_AUCs = []
        self.ACCURACYs = []
        self.F1s = []

        
        with open('./gen_keys/public_key.pkl', 'rb') as f:
            self.__public_key = pickle.load(f)

        with open('./gen_keys/private_key.pkl', 'rb') as f:
            self.__private_key = pickle.load(f)



    def encrypt_parameters(self, parameters):
        encrypted_params = [
            self.__public_key.encrypt(value) for param in parameters for value in param.flatten()
        ]
        enc_result = []
        index = 0
        for param in parameters:
            num_elements = param.size
            reshaped_array = np.array(encrypted_params[index:index + num_elements]).reshape(param.shape)
            enc_result.append(reshaped_array)
            index += num_elements
        return enc_result

    def decrypt_parameters(self, parameters):
        decrypted_params = [
            self.__private_key.decrypt(value) for param in parameters for value in param.flatten()
        ]
        dec_result = []
        index = 0
        for param in parameters:
            num_elements = param.size
            reshaped_array = np.array(decrypted_params[index:index + num_elements]).reshape(param.shape)
            dec_result.append(reshaped_array)
            index += num_elements
        return dec_result

    def get_parameters(self, config):
        print('================== INITIAL PARAMS ==================')
        params = ts.get_model_parameters(self.model)
        # encrypted_params = self.encrypt_parameters(params)
        # print(params)
        return params

    def fit(self, parameters, config):
        print('============================ PARAMS BEFORE FIT ===========================')
        print(parameters)
        
        # decrypted_params = self.decrypt_parameters(parameters)
        ts.set_model_parameters(self.model, parameters)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        
        print(f"Training finished for round {config.get('server_round', 'unknown')}")
        print('============================= PARAMETERS AFTER FIT ===============================')
        params_1 = ts.get_model_parameters(self.model)
        print(f'clear: {params_1}')
        encrypted_params = self.encrypt_parameters(params_1)
        print(f'Encrypted: {encrypted_params}')
        
        return encrypted_params, len(self.X_train), {}

    def evaluate(self, parameters, config):
        print('========================== EVALUATE PARAMS =============================================')
        print(parameters, parameters[0].size, parameters[1].size)
        
        parameters = self.decrypt_parameters(parameters)
        print(f'Decrypted for EVAL {parameters}')
        
        ts.set_model_parameters(self.model, parameters)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = self.model.predict(self.X_test)
        
        loss = log_loss(self.y_test, y_pred_proba)
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f'accuracy: {accuracy}')
        print(f'ROC_AUC: {roc_auc}')
        print(f'f1_score: {f1}')
        
        self.losses.append(loss)
        self.ROC_AUCs.append(roc_auc)
        self.ACCURACYs.append(accuracy)
        self.F1s.append(f1)
        
        metrics = {"accuracy": accuracy, "roc_auc": roc_auc, "f1-score": f1}
        print(f"Metrics: {metrics}")
        
        return loss, len(self.X_test), metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flower client with specific version")
    parser.add_argument("--strategy-id", type=int, required=True, help="strategy ID to determine ")
    args = parser.parse_args()

    
    if args.strategy_id == 0:
        strategy_suffix = "fed_avg"
    elif args.strategy_id == 1:
        strategy_suffix = "dp_fixed"
    elif args.strategy_id == 2:
        strategy_suffix = "dp_adaptive"
    elif args.strategy_id == 3:
        strategy_suffix = "fed_avg_paillier"
    else:
        raise ValueError("Unsupported strategy. Use 0 for fed_avg, 1 for dp_fixed, or 2 for dp_adaptive")

    
    model = LogisticRegression(
        max_iter=1,  
        warm_start=True
    )

    scaler = MinMaxScaler()
    smote = SMOTE(random_state=42)

    path_for_train_data = './datas/IID_df_1.csv'
    path_for_test_data = './datas/test_glob.csv'


    data_train = pd.read_csv(path_for_train_data)
    data_test = pd.read_csv(path_for_test_data)

    X_train_, y_train_ = data_train.drop(columns='Fraud'), data_train['Fraud']
    X_test_, y_test = data_test.drop(columns='Fraud'), data_test['Fraud']

    
    X_train_scale = scaler.fit_transform(X_train_)
    X_train, y_train = smote.fit_resample(X_train_scale, y_train_)
    X_test = scaler.transform(X_test_)


    ts.set_initial_parameters(model)

    client = CustomClient(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,

        )


 

    fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=client
        )

   

    ts.save_metrics_json(client, strategy_suffix)



