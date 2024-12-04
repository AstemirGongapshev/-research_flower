import pickle
import os
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from typing import Tuple
import pandas as pd
import numpy as np

def prepare_data(
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    batch_size: int = 32,
    save_dir: str = "./transformers"  
) -> Tuple[DataLoader, DataLoader, int]:
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in X_test.columns:
        X_test = X_test.drop(columns=['Unnamed: 0'])

    os.makedirs(save_dir, exist_ok=True)  

    try:
        X = df.drop(columns="Fraud")
        y = df["Fraud"]


        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)

 
        with open(os.path.join(save_dir, "scaler_noniid.pkl"), "wb") as f:
            pickle.dump(scaler, f)



        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)


        with open(os.path.join(save_dir, "poly_noniid.pkl"), "wb") as f:
            pickle.dump(poly, f)

        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_poly, y)


        
        X_train_tensor = torch.from_numpy(X_train_resampled.astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train_resampled.to_numpy().astype(np.int64))

        X_test_tensor = torch.from_numpy(X_test_poly.astype(np.float32))
        y_test_tensor = torch.from_numpy(y_test.to_numpy().astype(np.int64))

 
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_dim = X_train_tensor.shape[1]
        return train_loader, test_loader, input_dim

    except Exception as e:
        logging.error(f"Failed to prepare data: {e}")
        raise






prepare_data(
    pd.read_csv("./datas/NON_IID_FL_1.csv"), 
    (pd.read_csv("./datas/TEST_SAMPLE.csv").drop(columns="Fraud")),
    (pd.read_csv("./datas/TEST_SAMPLE.csv")).Fraud,
)


