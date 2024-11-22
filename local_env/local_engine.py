
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

if __name__ == "__main__":
    
    test_sample = get_data(TEST_SAMPLE_PATH)

    #TODO IID
    glob_iid = get_data(GLOBAL_DATA_IID_PATH)
    X_global_train, y_global_train, X_test_iid = prepare_data(glob_iid, test_sample.drop(columns="Fraud"))
    _, glob_metrics_iid = fit_predict(X_global_train, X_test_iid, y_global_train, test_sample.Fraud)
    save_results(glob_metrics_iid, key_iid, PATH_FOR_IID_RES)

    #TODO NON_IID
    glob_non_iid = get_data(GLOBAL_DATA_NON_IID_PATH)
    columns = glob_non_iid.drop(columns="Fraud").columns
    glob_non_iid_noise = add_gaussian_noise(glob_non_iid, columns, noise_level=1.5)
    X_global_train_n, y_global_train_n, X_test_non_iid = prepare_data(glob_non_iid_noise, test_sample.drop(columns="Fraud"))
    _, glob_metrics_non_iid = fit_predict(X_global_train_n, X_test_non_iid, y_global_train_n, test_sample.Fraud)
    save_results(glob_metrics_non_iid, key_noniid, PATH_FOR_NON_IID_RES)
 

    # TODO client_1 iid, client_2 iid
    client_1_iid = get_data(CLIENT_1_DATA_IID_PATH)
    X_train_1, y_train_1, X_test_iid_1 = prepare_data(client_1_iid, test_sample.drop(columns="Fraud"))
    _, client1_metrics_iid = fit_predict(X_train_1, X_test_iid_1, y_train_1, test_sample.Fraud)
    save_results(client1_metrics_iid, key_1_iid, PATH_FOR_IID_RES)

    client_2_iid = get_data(CLIENT_2_DATA_IID_PATH)
    X_train_2, y_train_2, X_test_iid_2 = prepare_data(client_2_iid, test_sample.drop(columns="Fraud"))
    _, client2_metrics_iid = fit_predict(X_train_2, X_test_iid_2, y_train_2, test_sample.Fraud)
    save_results(client2_metrics_iid, key_2_iid, PATH_FOR_IID_RES)


    # TODO NON_IID client_1, client_2 with normal noise 
    client_1_non_iid = get_data(CLIENT_1_DATA_NON_IID_PATH)
    columns_1 = client_1_non_iid.drop(columns="Fraud").columns
    client_1_non_iid_noise = add_gaussian_noise(client_1_non_iid, columns_1, noise_level=1.5)

    X_train_1_non, y_train_1_non, X_test_non_iid_1 = prepare_data(client_1_non_iid_noise, test_sample.drop(columns="Fraud"))
    _, client1_metrics_non_iid = fit_predict(X_train_1_non, X_test_non_iid_1, y_train_1_non, test_sample.Fraud)
    save_results(client1_metrics_non_iid, key_1_noniid, PATH_FOR_NON_IID_RES)



    client_2_non_iid = get_data(CLIENT_2_DATA_NON_IID_PATH)
    columns_2 = client_2_non_iid.drop(columns="Fraud").columns
    client_2_non_iid_noise = add_gaussian_noise(client_2_non_iid, columns_2, noise_level=1.5)
    
    X_train_2_non, y_train_2_non, X_test_non_iid_2 = prepare_data(client_2_non_iid_noise, test_sample.drop(columns="Fraud"))
    _, client2_metrics_non_iid = fit_predict(X_train_2_non, X_test_non_iid_2, y_train_2_non, test_sample.Fraud)
    save_results(client2_metrics_non_iid, key_2_noniid, PATH_FOR_NON_IID_RES)

    
