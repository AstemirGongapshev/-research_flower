import os
import numpy as np
import pandas as pd
import torch
import json
import torch.nn as nn
from typing import List, Tuple, Dict
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from datetime import datetime




log_dir = "./fed_env/process"
os.makedirs(log_dir, exist_ok=True)


log_filename = os.path.join(log_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)








def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """
    Retrieve the parameters of a PyTorch model as a list of numpy arrays.

    Parameters:
    model (torch.nn.Module): The PyTorch model.

    Returns:
    List[np.ndarray]: A list containing the model's parameters.
    """
    return [param.data.cpu().numpy() for param in model.parameters()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Set the parameters of a PyTorch model from a list of numpy arrays.

    Parameters:
    model (torch.nn.Module): The PyTorch model.
    parameters (List[np.ndarray]): A list containing the model's parameters.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict)


def set_initial_parameters(model: torch.nn.Module) -> None:
    """
    Initialize a PyTorch model's parameters with random values.

    Parameters:
    model (torch.nn.Module): The PyTorch model.
    """
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)


def prepare_data(
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Preprocesses the data by scaling features, generating polynomial features, applying SMOTE,
    and creating DataLoaders.

    Parameters:
    - df (pd.DataFrame): Training data with a 'Fraud' column as the target variable.
    - X_test (pd.DataFrame): Test data to be scaled and transformed.
    - y_test (pd.Series): Test target vector.
    - random_state (int): Random seed for reproducibility.
    - batch_size (int): Batch size for DataLoaders.

    Returns:
    - Tuple[DataLoader, DataLoader, int]:
        - train_loader (DataLoader): DataLoader for the training set.
        - test_loader (DataLoader): DataLoader for the test set.
        - input_dim (int): Input dimension for the model.
    """
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in X_test.columns:
        X_test = X_test.drop(columns=['Unnamed: 0'])

    try:
        X = df.drop(columns="Fraud")
        y = df["Fraud"]

        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)

        logging.info("Features scaled using MinMaxScaler.")

        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        logging.info("Polynomial features of degree 2 generated.")

       
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_poly, y)

        logging.info(f"SMOTE applied. Training sample size increased to {len(X_train_resampled)}.")

        
        X_train_tensor = torch.from_numpy(X_train_resampled.astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train_resampled.to_numpy().astype(np.int64))

        X_test_tensor = torch.from_numpy(X_test_poly.astype(np.float32))
        y_test_tensor = torch.from_numpy(y_test.to_numpy().astype(np.int64))

      
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logging.info("DataLoaders for Train and Test sets created.")

        input_dim = X_train_tensor.shape[1]
        return train_loader, test_loader, input_dim

    except Exception as e:
        logging.error(f"Failed to prepare data: {e}")
        raise



def save_metrics_json(client, strategy_suffix: str, filename: str) -> None:
    """
    Save client metrics to a JSON file. Append metrics to the given strategy suffix.

    Parameters:
    client: An object containing a list of metrics.
    strategy_suffix (str): The key under which metrics are stored (e.g., 'fed_avg_noniid').
    filename (str): Path to the JSON file where metrics are saved.
    """
   
    metrics_list = client.metrics  

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

    
    all_metrics[strategy_suffix].extend(metrics_list)

   
    os.makedirs(os.path.dirname(filename), exist_ok=True)

   
    with open(filename, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    print(f"Metrics successfully saved to {filename} under suffix {strategy_suffix}")




def train(model: torch.nn.Module, train_loader: DataLoader, learning_rate: float, num_epochs: int, device: str) -> None:
    """
    Train the PyTorch model using the provided DataLoader.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to train.
    - train_loader (DataLoader): DataLoader containing the training data.
    - learning_rate (float): Learning rate for the optimizer.
    - num_epochs (int): Number of epochs to train the model.
    - device (str): Device to use for training ('cpu' or 'cuda').
    """
    try:
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logging.info(f"Epoch {epoch + 1}/{num_epochs}: Loss = {epoch_loss:.4f}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


def test(model: torch.nn.Module, test_loader: DataLoader, device: str) -> Dict[str, float]:
    """
    Evaluate the PyTorch model using the provided DataLoader.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to evaluate.
    - test_loader (DataLoader): DataLoader containing the test data.
    - device (str): Device to use for evaluation ('cpu' or 'cuda').

    Returns:
    - Dict[str, float]: Dictionary containing evaluation metrics (log-loss, ROC AUC, accuracy, F1 score).
    """
    model.to(device)
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

    try:
        test_logloss = log_loss(test_labels, test_predictions)
        test_roc_auc = roc_auc_score(test_labels, test_predictions)
        test_pred_binary = np.array(test_predictions) > 0.5
        test_accuracy = accuracy_score(test_labels, test_pred_binary)
        test_f1 = f1_score(test_labels, test_pred_binary)

        logging.info(
            f"Test Metrics | Log-loss: {test_logloss:.4f} | "
            f"ROC AUC: {test_roc_auc:.4f} | Accuracy: {test_accuracy:.4f} | F1 Score: {test_f1:.4f}"
        )

        return {
            "logloss_test": test_logloss,
            "roc_auc_test": test_roc_auc,
            "accuracy_test": test_accuracy,
            "f1_test": test_f1,
        }

    except ValueError as e:
        logging.error(f"Error in metric computation: {e}")
        return {}