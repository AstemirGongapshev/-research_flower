import flwr as fl
import torch

from typing import Dict, Tuple
from flwr.common import NDArrays

from fed_tasks_torch import (

    prepare_data,
    set_initial_parameters,
    get_model_parameters,
    set_model_parameters,
    train,
    test,
    save_metrics_json
)

from model import LogisticRegressionModel

from fed_tasks_sklearn import get_data
import logging
from datetime import datetime
import os



glob_round = 0


class CustomClient(fl.client.NumPyClient):
    def __init__(self, model: torch.nn.Module, train_loader, test_loader, device: str):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.metrics = []

    def get_parameters(self, config: Dict[str, int]) -> NDArrays:

        return get_model_parameters(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, int]) -> Tuple[NDArrays, int, Dict]:

        global glob_round
        glob_round += 1
       
        set_model_parameters(self.model, parameters)

        train(self.model, self.train_loader, learning_rate=0.001, num_epochs=1, device=self.device)

     
        return get_model_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, int]) -> Tuple[float, int, Dict]:

        global glob_round
        set_model_parameters(self.model, parameters)

        
        metrics = test(self.model, self.test_loader, device=self.device)

        
        # logging.info(f"Evaluation Metrics (Round {glob_round}):")
        # for key, value in metrics.items():
        #     logging.info(f" - {key.capitalize()}: {value:.4f}")

        self.metrics.append(metrics)

        
        return metrics["logloss_test"], len(self.test_loader.dataset), metrics


if __name__ == "__main__":
  
    TRAIN_DATA_PATH = "./datas/IID_1.csv"
    TEST_SAMPLE_PATH = "./datas/TEST_SAMPLE.csv"
    SAVE_PATH = "./fed_env/results/iid.json"


    data_noniid = get_data(TRAIN_DATA_PATH)
    data_test = get_data(TEST_SAMPLE_PATH)

   
    train_loader, test_loader, input_dim = prepare_data(
        data_noniid,
        data_test.drop(columns="Fraud"),
        data_test["Fraud"],
        batch_size=64
    )

  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LogisticRegressionModel(input_dim=input_dim)
    set_initial_parameters(model)

   
    client = CustomClient(model, train_loader, test_loader, device)

    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


    # save_metrics_json(client, "fed_avg_iid", SAVE_PATH)
