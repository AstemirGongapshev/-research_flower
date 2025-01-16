
from typing import Dict
import flwr as fl
from strategy import EarlyStoppingFedAvg  
import torch
from fed_tasks_torch import prepare_validation_data, load_transformers
from model import LogisticRegressionModel
import logging
import os
from datetime import datetime

# log_dir = "./fed_env/process"
# os.makedirs(log_dir, exist_ok=True)

# log_filename = os.path.join(log_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
# logging.basicConfig(
#     filename=log_filename,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

def fit_round(config):
    return {"server_round": config.num_rounds}

if __name__ == "__main__":
    TRANSFORMER_PATH = "./transformers"

   
    scaler_iid, poly_iid = load_transformers(TRANSFORMER_PATH, key="noniid")

   
    val_loader, input_dim_val, test_loader, _ = prepare_validation_data(
        scaler=scaler_iid,
        poly=poly_iid
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LogisticRegressionModel(input_dim=input_dim_val).to(device)


    fed_strategy = EarlyStoppingFedAvg(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        patience=30,  
        on_fit_config_fn=fit_round,  
    )

    ServerHistory=fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=fed_strategy,
        config=fl.server.ServerConfig(num_rounds=150),  
    )
    # TODO implement validation for centralized metrics>> can cath in ServerConfig.metrics_centralized


    # ServerHistory.losses_distributed[0][0]
    # logging.info(f'Centralized metrics: {ServerHistory.metrics_centralized}')
    # logging.info(f'Num rounds: {')
    
    fed_strategy.save_metrics(fed_strategy.round_metrics, filename="./fed_env/results/noniid.json", strategy_suffix="fed_prox(mu=0.3)")

    # logging.info("Training stopped. Early stopping or all rounds are completed.")
    print("Training has stopped. Metrics saved.")
