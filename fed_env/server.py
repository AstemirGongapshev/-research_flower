from typing import Dict
import flwr as fl
from strategy import EarlyStoppingFedAvg  
import torch 
from fed_tasks_torch import prepare_validation_data, load_transformers
from model import LogisticRegressionModel
import logging
import os
from datetime import datetime



log_dir = "./fed_env/process"
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(log_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def fit_round(server_round):
    
    return {"server_round": server_round}


if __name__ == "__main__":
    
    TRANSFORMER_PATH = "./transformers"

    
    scaler_iid, poly_iid = load_transformers(TRANSFORMER_PATH, key="iid")

    
    val_loader, input_dim_val,  _1, _2 = prepare_validation_data(scaler=scaler_iid, poly=poly_iid)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LogisticRegressionModel(input_dim=input_dim_val).to(device)


    fed_strategy = EarlyStoppingFedAvg(
        model=model,
        val_loader=val_loader,
        device=device,
        patience=3, 
        on_fit_config_fn=fit_round,
    )

    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=fed_strategy,
        config=fl.server.ServerConfig(num_rounds=50), 
    )



#     # dp_strategy = DifferentialPrivacyServerSideFixedClipping(
#     #     strategy=fed_strategy,
#     #     noise_multiplier=5.0,
#     #     clipping_norm=1.0,
#     #     num_sampled_clients=2,
#     # )

#     # dp_strategy = DifferentialPrivacyServerSideAdaptiveClipping(
#     #     strategy=fed_strategy, 
#     #     noise_multiplier=5.0,
#     #     num_sampled_clients=2,
#     #     initial_clipping_norm=1.0,
#     #     target_clipped_quantile=0.8,
#     #     clipped_count_stddev=10.0  # Попробуйте увеличить это значение
#     # )