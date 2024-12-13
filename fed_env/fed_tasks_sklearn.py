import numpy as np
import os
import json
import logging
from typing import List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


log_dir = "./fed_env/process"
os.makedirs(log_dir, exist_ok=True)


log_filename = os.path.join(log_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)




def get_model_parameters(model) -> List[np.ndarray]:

    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_]
    return params


def set_model_parameters(model, params: List[np.ndarray]) -> object:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_parameters(model) -> None:
    n_features = 594  # Fixed number of features
    model.classes_ = np.arange(1)  # Dummy class
    model.coef_ = np.random.randn(1, n_features)
    if model.fit_intercept:
        model.intercept_ = np.random.randn(1)


def save_metrics_json(client, strategy_suffix: str, filename: str ) -> None:
    metrics = {
        "losses": list(client.losses),
        "ROC_AUCs": list(client.ROC_AUCs),
        "ACCURACYs": list(client.ACCURACYs),
        "F1s": list(client.F1s)
    }

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as f:
            try:
                all_metrics = json.load(f)
            except json.JSONDecodeError:
                all_metrics = {}
    else:
        all_metrics = {}

    if strategy_suffix not in all_metrics:
        all_metrics[strategy_suffix] = []

    all_metrics[strategy_suffix].append(metrics)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics successfully saved to {filename} under suffix {strategy_suffix}")


def get_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Data successfully loaded from {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        raise


def prepare_data(df: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in X_test.columns:
        X_test = X_test.drop(columns=['Unnamed: 0'])
    try:
        scaler = MinMaxScaler()
        smote = SMOTE(random_state=22)

        X, y = df.drop(columns="Fraud"), df.Fraud
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.2, random_state=234)

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_val_poly = poly.transform(X_val_scaled)
        X_test_poly = poly.transform(X_test_scaled)
        X_smote, y_smote = smote.fit_resample(X_train_poly, y_train)

        logging.info("Data successfully prepared (scaling, SMOTE, polynomial features).")
        return X_smote, X_val_poly, X_test_poly, y_smote, y_val
    except Exception as e:
        logging.error(f"Failed to prepare data: {e}")
        raise


