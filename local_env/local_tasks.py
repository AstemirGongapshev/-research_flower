import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE

from models import LogisticRegressionModel




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



def prepare_data(
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    validation_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Preprocesses the data by scaling features, generating polynomial features, applying SMOTE,
    splitting into training and validation sets, and creating DataLoaders.

    Parameters:
    - df (pd.DataFrame): Training data with a 'Fraud' column as the target variable.
    - X_test (pd.DataFrame): Test data to be scaled and transformed.
    - y_test (pd.Series): Test target vector.
    - validation_size (float): Proportion of the training data to include in the validation set.
    - random_state (int): Random seed for reproducibility.
    - batch_size (int): Batch size for DataLoaders.

    Returns:
    - Tuple[DataLoader, DataLoader, DataLoader]:
        - train_loader (DataLoader): DataLoader for the training set.
        - val_loader (DataLoader): DataLoader for the validation set.
        - test_loader (DataLoader): DataLoader for the test set.
    """
   
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in X_test.columns:
        X_test = X_test.drop(columns=['Unnamed: 0'])

    try:
        
        X = df.drop(columns="Fraud")
        y = df["Fraud"]

        
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=validation_size,
            random_state=random_state,
            stratify=y
        )

        logging.info(f"Data split into Train ({len(X_train)}) and Val ({len(X_val)}) samples.")


        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        logging.info("Features scaled using MinMaxScaler.")

        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_val_poly = poly.transform(X_val_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        logging.info("Polynomial features of degree 2 generated.")

        
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_poly, y_train)

        logging.info(f"SMOTE applied. Training sample size increased to {len(X_train_resampled)}.")

        
        X_train_tensor = torch.from_numpy(X_train_resampled.astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train_resampled.to_numpy().astype(np.int64))

        X_val_tensor = torch.from_numpy(X_val_poly.astype(np.float32))
        y_val_tensor = torch.from_numpy(y_val.to_numpy().astype(np.int64))

        X_test_tensor = torch.from_numpy(X_test_poly.astype(np.float32))
        y_test_tensor = torch.from_numpy(y_test.to_numpy().astype(np.int64))

        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logging.info("DataLoaders for Train, Val, and Test sets created.")

        return train_loader, val_loader, test_loader

    except Exception as e:
        logging.error(f"Failed to prepare data: {e}")
        raise




def fit_predict(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    device: torch.device = torch.device('cpu'),
    learning_rate: float = 0.001,
    num_epochs: int = 50,
    patience: int = 5
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Trains a logistic regression model and evaluates its performance.

    Parameters:
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    - test_loader (DataLoader): DataLoader for test data.
    - input_dim (int): Dimensionality of the input data.
    - device (torch.device): Device for training (CPU or GPU).
    - learning_rate (float): Learning rate for the optimizer.
    - num_epochs (int): Maximum number of training epochs.
    - patience (int): Number of epochs with no improvement in ROC AUC before early stopping.

    Returns:
    - Tuple[np.ndarray, Dict[str, float]]:
        - coefficients (np.ndarray): Model coefficients.
        - metrics (Dict[str, float]): Dictionary containing evaluation metrics (log-loss, ROC AUC, accuracy, F1 score).
    """
    try:
        
        model = LogisticRegressionModel(input_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_val_roc_auc = 0.0
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

           
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.squeeze().long())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)

            
            model.eval()
            val_loss = 0.0
            val_labels = []
            val_predictions = []

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels.squeeze().long())
                    val_loss += loss.item()

                    probabilities = torch.softmax(outputs, dim=1)[:, 1]
                    val_labels.extend(labels.cpu().numpy())
                    val_predictions.extend(probabilities.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_roc_auc = roc_auc_score(val_labels, val_predictions)
            val_pred_binary = np.array(val_predictions) > 0.5
            val_accuracy = accuracy_score(val_labels, val_pred_binary)
            val_f1 = f1_score(val_labels, val_pred_binary)

            logging.info(
                f'Epoch [{epoch+1}/{num_epochs}] | '
                f'Train Loss: {avg_epoch_loss:.4f} | '
                f'Val Loss: {avg_val_loss:.4f} | '
                f'Val ROC AUC: {val_roc_auc:.4f} | '
                f'Val Accuracy: {val_accuracy:.4f} | '
                f'Val F1 Score: {val_f1:.4f}'
            )

          
            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
                logging.info(f'ROC AUC improved to {val_roc_auc:.4f}. Saving model.')
            else:
                epochs_without_improvement += 1
                logging.info(f'ROC AUC did not improve. {epochs_without_improvement}/{patience} epochs without improvement.')

           
            if epochs_without_improvement >= patience:
                logging.info(f'Early stopping triggered at epoch {epoch+1}.')
                break

        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logging.info('Best model state loaded based on validation ROC AUC.')

       
        model.eval()
        test_labels = []
        test_predictions = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                test_labels.extend(labels.cpu().numpy())
                test_predictions.extend(probabilities.cpu().numpy())

        
        test_logloss = log_loss(test_labels, test_predictions)
        test_roc_auc = roc_auc_score(test_labels, test_predictions)
        test_pred_binary = np.array(test_predictions) > 0.5
        test_accuracy = accuracy_score(test_labels, test_pred_binary)
        test_f1 = f1_score(test_labels, test_pred_binary)

        logging.info(
            f'Test Metrics | Log-loss: {test_logloss:.4f} | '
            f'ROC AUC: {test_roc_auc:.4f} | Accuracy: {test_accuracy:.4f} | F1 Score: {test_f1:.4f}'
        )

        
        coefficients = model.fc.weight.detach().cpu().numpy()[1]

        metrics = {
            "logloss_test": test_logloss,
            "roc_auc_test": test_roc_auc,
            "accuracy_test": test_accuracy,
            "f1_test": test_f1,
        }

        logging.info("Model training and evaluation completed successfully.")
        return coefficients, metrics

    except Exception as e:
        logging.error(f"Training or evaluation failed: {e}")
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

