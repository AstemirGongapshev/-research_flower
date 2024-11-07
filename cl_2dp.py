import flwr as fl 
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tasks as ts
import warnings
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt





class CustomClientDP(fl.client.NumPyClient):
  
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
         
        return ts.get_model_parameters(self.model)
    
    def fit(self, parameters, config):
        ts.set_model_parameters(self.model, parameters)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train, y_train)
    
        
        return ts.get_model_parameters(self.model), len(X_train), {}

    def evaluate(self, parameters, config):
        
        ts.set_model_parameters(self.model, parameters)
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
       
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
    
    N_CLIENTS = 2

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.partition_id

    df = pd.read_csv(f'./IID_df_{partition_id+1}.csv')
    testing = pd.read_csv('./test_glob.csv')
    scaler = MinMaxScaler()
    smote = SMOTE(random_state=42)



    X_, y_ = df.drop(columns="Fraud"), df.Fraud
    X_test_, y_test = testing.drop(columns='Fraud'), testing.Fraud 
    X_train_scale = scaler.fit_transform(X_)
    X_train, y_train = smote.fit_resample(X_train_scale, y_)

    X_test = scaler.transform(X_test_)


    model = LogisticRegression(
        max_iter=1,
        warm_start=True
    )

    ts.set_initial_parameters(model)
    
    
    client_dp = CustomClientDP(model, X_train, X_test, y_train, y_test)



    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client_dp
    )


    