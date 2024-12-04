import torch
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Dict, Union, Optional
from flwr.common import Parameters, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
import logging



class EarlyStoppingFedAvg(FedAvg):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        val_loader,  
        device: str = "cpu",
        patience: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.best_roc_auc = -float("inf")
        self.counter = 0
        self.should_stop = False
        self.global_round = 0
        logging.info("EarlyStoppingFedAvg initialized.")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Union[Tuple, BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is None:
            return None, {}

        
        self.set_model_parameters_from_aggregated(aggregated_params)

        
        roc_auc = self.evaluate_on_validation()

        
        if roc_auc > self.best_roc_auc:
            self.best_roc_auc = roc_auc
            self.counter = 0
        else:
            self.counter += 1
            logging.info(f"No improvement in ROC AUC for {self.counter} round(s).")

        if self.counter >= self.patience:
            logging.warning("Early stopping triggered!")
            self.should_stop = True

        
        logging.info(f"Round {server_round}: ROC AUC = {roc_auc:.4f}")
        return aggregated_params, aggregated_metrics

    def set_model_parameters_from_aggregated(self, parameters: Parameters):
        """Устанавливаем параметры в модель из агрегированных данных."""
        param_arrays = parameters_to_ndarrays(parameters)
        state_dict = {
            key: torch.tensor(value).to(self.device)
            for key, value in zip(self.model.state_dict().keys(), param_arrays)
        }
        self.model.load_state_dict(state_dict)

    def evaluate_on_validation(self) -> float:
        """Оценка модели на валидационной выборке."""
        self.model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(probabilities.cpu().numpy())

        
        roc_auc = roc_auc_score(all_labels, all_predictions)
        return roc_auc
