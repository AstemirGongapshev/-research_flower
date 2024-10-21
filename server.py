import flwr as fl
import time as time

from flwr.common import parameters_to_ndarrays

def fit_round(server_round):
    
    return {"server_round": server_round}



def print_parameters(parameters, stage):
    
    param_arrays = parameters_to_ndarrays(parameters)
    print(f"Параметры {stage}:")
    for idx, param in enumerate(param_arrays):
        print(f"Параметр {idx}: {param[:5]}...") 


class CustomFedAvg(fl.server.strategy.FedAvg):
    

    def configure_fit(self, server_round, parameters, client_manager):
       
        print_parameters(parameters, f"до агрегации в раунде {server_round}")
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
       
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        
        if aggregated_parameters:
            print_parameters(aggregated_parameters, f"после агрегации в раунде {server_round}")
        
        return aggregated_parameters, metrics_aggregated


if __name__ == "__main__":

    strategy = CustomFedAvg(
        min_available_clients=2,
        on_fit_config_fn=fit_round,
    )
    
    start_time = time.time()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=4)
    )

    end_time = time.time()
    ex_time = (end_time - start_time)
    print(f"Общее время выполнения: {ex_time} секунд")
