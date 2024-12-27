import pandas as pd
from local_tasks import (
    get_data,
    prepare_data,
    add_gaussian_noise,
    fit_predict,
    save_results
)

CLIENT_1_DATA_IID_PATH = "./datas/IID_1.csv"
CLIENT_2_DATA_IID_PATH = "./datas/IID_2.csv"
GLOBAL_DATA_IID_PATH = "./datas/IID.csv"

CLIENT_1_DATA_NON_IID_PATH = "./datas/NON_IID_1.csv"
CLIENT_2_DATA_NON_IID_PATH = "./datas/NON_IID_2.csv"
GLOBAL_DATA_NON_IID_PATH = "./datas/NON_IID.csv"

TEST_SAMPLE_PATH = "./datas/TEST_SAMPLE.csv"

PATH_FOR_IID_RES = "./local_env/results/iid_res.json"
PATH_FOR_NON_IID_RES = "./local_env/results/non_iid_res.json"


key_1_iid = "local_1_iid"
key_2_iid = "local_2_iid"
key_iid = "global_iid"

key_1_noniid = "local_1_noniid"
key_2_noniid = "local_2_noniid"
key_noniid = "global_noniid"

def process_dataset(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_labels: pd.Series,
    add_noise: bool,
    noise_level: float,
    key: str,
    save_path: str
):

    try:
        if add_noise:
            columns = df.drop(columns="Fraud").columns
            df = add_gaussian_noise(df, columns, noise_level=noise_level)
            print(f"Added Gaussian noise with level {noise_level}.")

        
        train_loader, val_loader, test_loader = prepare_data(
            df,
            test_df,
            test_labels
        )

        
        for X_batch, y_batch in train_loader:
            input_dim = X_batch.shape[1]
            break


        
        _, metrics = fit_predict(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            input_dim=input_dim
        )

        
        save_results(metrics, key, save_path)
        print(f"Results saved with key '{key}' in '{save_path}'.")
    
    except Exception as e:
        print(f"Error processing dataset with key '{key}': {e}")

def main():
    # Load the test sample dataset
    test_sample = get_data(TEST_SAMPLE_PATH)
    X_test = test_sample.drop(columns="Fraud")
    y_test = test_sample.Fraud

    # --- Global IID ---
    glob_iid = get_data(GLOBAL_DATA_IID_PATH)
    process_dataset(
        df=glob_iid,
        test_df=X_test,
        test_labels=y_test,
        add_noise=False,
        noise_level=0.0,  # Not used since add_noise=False
        key=key_iid,
        save_path=PATH_FOR_IID_RES
    )

    # --- Global NON_IID ---
    glob_non_iid = get_data(GLOBAL_DATA_NON_IID_PATH)
    process_dataset(
        df=glob_non_iid,
        test_df=X_test,
        test_labels=y_test,
        add_noise=True,
        noise_level=1.5,
        key=key_noniid,
        save_path=PATH_FOR_NON_IID_RES
    )

    # --- Client 1 IID ---
    client_1_iid = get_data(CLIENT_1_DATA_IID_PATH)
    process_dataset(
        df=client_1_iid,
        test_df=X_test,
        test_labels=y_test,
        add_noise=False,
        noise_level=0.0,
        key=key_1_iid,
        save_path=PATH_FOR_IID_RES
    )

    # --- Client 2 IID ---
    client_2_iid = get_data(CLIENT_2_DATA_IID_PATH)
    process_dataset(
        df=client_2_iid,
        test_df=X_test,
        test_labels=y_test,
        add_noise=False,
        noise_level=0.0,
        key=key_2_iid,
        save_path=PATH_FOR_IID_RES
    )

    # --- Client 1 NON_IID ---
    client_1_non_iid = get_data(CLIENT_1_DATA_NON_IID_PATH)
    process_dataset(
        df=client_1_non_iid,
        test_df=X_test,
        test_labels=y_test,
        add_noise=True,
        noise_level=1.5,
        key=key_1_noniid,
        save_path=PATH_FOR_NON_IID_RES
    )

    # --- Client 2 NON_IID ---
    client_2_non_iid = get_data(CLIENT_2_DATA_NON_IID_PATH)
    process_dataset(
        df=client_2_non_iid,
        test_df=X_test,
        test_labels=y_test,
        add_noise=True,
        noise_level=1.5,
        key=key_2_noniid,
        save_path=PATH_FOR_NON_IID_RES
    )

if __name__ == "__main__":
    main()
