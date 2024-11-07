import flwr as fl
import time
from flwr.server.strategy import FedAvg
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping

def fit_round(server_round):
    """Отправляет номер раунда клиенту."""
    return {"server_round": server_round}

if __name__ == "__main__":
    # Базовая стратегия FedAvg
    fed_strategy = FedAvg(
        min_available_clients=2,
        on_fit_config_fn=fit_round,
    )
    
    # # Оберните стратегию в DifferentialPrivacyClientSideFixedClipping
    # dp_strategy = DifferentialPrivacyClientSideFixedClipping(
    #     strategy=fed_strategy,
    #     noise_multiplier=0.5,
    #     clipping_norm=0.1,
    #     num_sampled_clients=2,
    # )

    start_time = time.time()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=fed_strategy,
        config=fl.server.ServerConfig(num_rounds=10)
    )

    end_time = time.time()
    ex_time = (end_time - start_time)
    print(f"Общее время выполнения: {ex_time} секунд")
