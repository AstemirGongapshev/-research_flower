import os 
import json
import pickle
import logging
import warnings
import argparse
import flwr as fl 
import numpy as np
import tasks as ts
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from flwr.common import NDArrays
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

    def get_parameters(self, config):
         
        return ts.get_model_parameters(model)
    
    def fit(self, parameters, config):
        print(parameters)
        
        ts.set_model_parameters(model, parameters)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
    
        
        return ts.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):
        
        ts.set_model_parameters(model, parameters)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
       
        loss = log_loss(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        print(f'accuracy: {accuracy}')
        print(f'ROC_AUC: {roc_auc}')
        print(f'f1_score: {f1}')

        
        self.losses.append(loss)
        self.ROC_AUCs.append(roc_auc)
        self.ACCURACYs.append(accuracy)
        self.F1s.append(f1)
        
        
        return loss, len(X_test), {"accuracy": accuracy, "roc_auc": roc_auc, "f1-score": f1}


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
    else:
        raise ValueError("Unsupported strategy. Use 0 for fed_avg, 1 for dp_fixed, or 2 for dp_adaptive")

    N_CLIENTS = 2

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
    
    client_1 = CustomClient(model, X_train, X_test, y_train, y_test)
    
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client_1
    )

    
    ts.save_metrics_json(client_1, strategy_suffix)