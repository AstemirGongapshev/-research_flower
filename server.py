import flwr as fl
import time
from flwr.server.strategy import FedAvg
from flwr.server.strategy import ( DifferentialPrivacyServerSideFixedClipping,
                                   DifferentialPrivacyServerSideAdaptiveClipping,
                                   FedAvg
                                   ) 

def fit_round(server_round):
   
    return {"server_round": server_round}

if __name__ == "__main__":
   
    fed_strategy = FedAvg(
        min_available_clients=2,
        on_fit_config_fn=fit_round,
    )
    

    # dp_strategy = DifferentialPrivacyServerSideFixedClipping(
    #     strategy=fed_strategy,
    #     noise_multiplier=0.5,
    #     clipping_norm=0.1,
    #     num_sampled_clients=2,
    # )



    dp_stratagy = DifferentialPrivacyServerSideAdaptiveClipping(

        strategy=fed_strategy, 
        noise_multiplier=1.0,
        num_sampled_clients=2,
        initial_clipping_norm=0.1,
        target_clipped_quantile=0.5,
        clipped_count_stddev=1.0
    )
    

    start_time = time.time()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=dp_stratagy,
        config=fl.server.ServerConfig(num_rounds=50)
    )

    end_time = time.time()
    ex_time = (end_time - start_time)
    print(f"Общее время выполнения: {ex_time} секунд")
