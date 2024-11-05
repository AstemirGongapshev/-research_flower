import flwr as fl 
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import FedAvg.tasks as ts
import warnings
import argparse
import numpy as np


class CustomClient(fl.client.NumPyClient):
        

    def get_parameters(self, config):

        print('================== INITIAL PARAMS ==================')
        params = ts.get_model_parameters(model)
        print(params)

        return params

    def fit(self, parameters, config):
        
        
        ts.set_model_parameters(model, parameters)
        print('============================ PARAMS BEFORE  FIT===========================')
        print(parameters)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_smote_train, y_smote_train)
        print(f"Training finished for round {config['server_round']}")
        print('============================= PARAMETERS AFTER FIT ===============================')
        params_1 = ts.get_model_parameters(model)
        print(f'clear: {params_1}')



        return params_1, len(X_train_scale), {}

    def evaluate(self, parameters, config):
        losses = list()
        ROC_AUCs = list()
        ACCURACYs = list()
        F1s = list()
        print('========================== evaluate PARAMS =============================================')
        # i got agg parameters for server, here i have to decrypt them
        print(parameters, parameters[0].size, parameters[1].size)
    
        ts.set_model_parameters(model, parameters)
        y_pred_proba = model.predict_proba(X_test_scale)[:, 1]
        y_pred = model.predict(X_test_scale)
        loss = log_loss(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        print(f'accuracy: {accuracy}')
        print(f'ROC_AUC: {roc_auc}')
        print(f'f1_score: {f1}')
        losses.append(loss)
        ROC_AUCs.append(roc_auc)
        ACCURACYs.append(accuracy)
        F1s.append(f1)
        
        return loss, len(X_test_scale), {"accuracy": accuracy,
                                          "roc_auc": roc_auc,
                                          "f1-score": f1} 



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
    
    dataset_train = pd.read_csv(f'./IID_df_{partition_id+1}.csv')
    
    dataset_test = pd.read_csv(f'./test_glob.csv')

    

    X_train, y_train = dataset_train.drop(columns=['Fraud']), dataset_train['Fraud']
    X_test, y_test = dataset_test.drop(columns='Fraud'), dataset_test['Fraud']

    scaler = MinMaxScaler()
    smote = SMOTE(random_state=42)

    X_train_scale = scaler.fit_transform(X_train)
    X_smote_train, y_smote_train = smote.fit_resample(X_train_scale, y_train)
    X_test_scale = scaler.transform(X_test)


    model = LogisticRegression(
        penalty='l2',
        max_iter=5
    )

    ts.set_initial_parameters(model)

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=CustomClient()
    )

