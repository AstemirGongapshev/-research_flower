
from typing import Dict
import flwr as fl
from strategy import EarlyStoppingFedAvg  
import torch
from fed_tasks_torch import prepare_validation_data, load_transformers
from model import LogisticRegressionModel
import logging
import os
from flwr.server.strategy import FedProx
from datetime import datetime

# log_dir = "./fed_env/process"
# os.makedirs(log_dir, exist_ok=True)

# log_filename = os.path.join(log_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
# logging.basicConfig(
#     filename=log_filename,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

strategy = FedProx(
    
)