import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import List, Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE


log_dir = "./local_env/process"
os.makedirs(log_dir, exist_ok=True)


log_filename = os.path.join(log_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)



def get_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Parameters:
    path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: Loaded dataset as a DataFrame.
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"Data successfully loaded from {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        raise


def add_gaussian_noise(df: pd.DataFrame, columns: List[str], noise_level: float = 0.01) -> pd.DataFrame:
    """
    Add Gaussian noise to specified columns of a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (List[str]): List of column names to which noise should be added.
    noise_level (float): Standard deviation of the Gaussian noise (default is 0.01).

    Returns:
    pd.DataFrame: DataFrame with added noise.
    """
    try:
        for col in columns:
            noise = np.random.normal(0, noise_level, size=df[col].shape)
            df[col] += noise
        logging.info(f"Gaussian noise added to columns: {columns} with noise level: {noise_level}")
        return df
    except Exception as e:
        logging.error(f"Failed to add Gaussian noise: {e}")
        raise


def prepare_data(df: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data by scaling, generating polynomial features, and applying SMOTE.

    Parameters:
    df (pd.DataFrame): Training data with a 'Fraud' column as the target variable.
    X_test (pd.DataFrame): Test data to be scaled and transformed.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: Processed training features, 
    resampled training targets, and processed test features.
    """
    if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in X_test.columns:
            X_test = X_test.drop(columns=['Unnamed: 0'])
    try:
        scaler = MinMaxScaler()
        smote = SMOTE(random_state=22)

        X_train, y_train = df.drop(columns="Fraud"), df.Fraud

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)
        X_smote, y_smote = smote.fit_resample(X_train_poly, y_train)

        logging.info("Data successfully prepared (scaling, SMOTE, polynomial features).")
        return X_smote, y_smote, X_test_poly
    except Exception as e:
        logging.error(f"Failed to prepare data: {e}")
        raise


def fit_predict(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Train a logistic regression model and evaluate its performance.

    Parameters:
    X_train (np.ndarray): Training feature matrix.
    X_test (np.ndarray): Test feature matrix.
    y_train (np.ndarray): Training target vector.
    y_test (np.ndarray): Test target vector.

    Returns:
    Tuple[np.ndarray, Dict[str, float]]:
        - Model coefficients.
        - Dictionary containing evaluation metrics (log-loss, ROC AUC, accuracy, F1 score).
    """
    try:
        model = LogisticRegression(max_iter=250)
        #TODO change SGD optimazer
        roc_auc_cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")

        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        roc_auc_test = roc_auc_score(y_test, y_pred_proba)
        logloss_test = log_loss(y_test, y_pred_proba)
        accuracy_test = accuracy_score(y_test, y_pred)
        f1_test = f1_score(y_test, y_pred)
        params = model.coef_

        logging.info("Model successfully trained and predictions made.")
        return params, {
            "logloss_test": logloss_test,
            "roc_auc_test": roc_auc_test,
            "roc_auc_val": roc_auc_cv.mean(),
            "accuracy_test": accuracy_test,
            "f1_test": f1_test,
        }
    except Exception as e:
        logging.error(f"Model training or evaluation failed: {e}")
        raise


def save_results(metrics: dict, key: str, path: str):
    """
    Save or update metrics for a specific model key in a JSON file.

    Parameters:
    metrics (dict): The metrics to save or update.
    key (str): The key representing the model (e.g., 'local_1_iid').
    path (str): Path to the JSON file (e.g., 'iid_res.json').
    """
    try:
        if os.path.exists(path):

            with open(path, "r") as f:
                try:
                    data = json.load(f)  
                except json.JSONDecodeError:
                    logging.warning(f"File {path} is empty or invalid. Starting fresh.")
                    data = {} 
        else:

            data = {}
        data[key] = metrics

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        logging.info(f"Metrics for '{key}' successfully saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save metrics for '{key}' to {path}: {e}")
        raise

