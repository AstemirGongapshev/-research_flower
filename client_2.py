# import flwr as fl 
# import pandas as pd
# import pickle
# from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import tasks as ts
# import warnings
# import argparse
# import numpy as np


# class CustomClient(fl.client.NumPyClient):

        
#     def __init__(self):
        
#         with open('public_key.pkl', 'rb') as f:
#             self.__public_key = pickle.load(f)

#         with open('private_key.pkl', 'rb') as f:
#             self.__private_key = pickle.load(f)


#     def get_parameters(self, config):

#         print('================== INITIAL PARAMS ==================')
#         params = ts.get_model_parameters(model)
#         print(params)

#         return params

#     def fit(self, parameters, config):
        
        
#         ts.set_model_parameters(model, parameters)
#         print('============================ PARAMS BEFORE  FIT===========================')
#         print(parameters)
        
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             model.fit(X_train, y_train)
#         print(f"Training finished for round {config['server_round']}")
#         print('============================= PARAMETERS AFTER FIT ===============================')
#         params_1 = ts.get_model_parameters(model)
#         print(f'clear: {params_1}')
#         encrypted_params = [(self.__public_key.encrypt(value),) for param in params_1 for value in param.flatten()]
#         enc_result = []
#         index = 0

#         for param in params_1:
#                 num_elements = param.size
#                 reshaped_array = np.array(encrypted_params[index:index + num_elements]).reshape(param.shape)
#                 enc_result.append(reshaped_array)
#                 index += num_elements
#             # encrypt parameters here


#         print(f'Encrypted: {enc_result}')
        
#         return enc_result, len(X_train), {}

#     def evaluate(self, parameters, config):
#         print('========================== evaluate PARAMS =============================================')
#         # i got agg parameters for server, here i have to decrypt them
#         print(parameters, parameters[0].size, parameters[1].size)
#         decrypted_params = [(self.__private_key.decrypt(value),) for param in parameters for value in param.flatten()]
#         dec_res = []
#         index = 0
#         for param in parameters:
#                num_elements = param.size
#                reshaped_array = np.array(decrypted_params[index:index + num_elements]).reshape(param.shape)
#                dec_res.append(reshaped_array)
#                index += num_elements
#         print(f' Decrypted for EVAL {dec_res}')


        
#         ts.set_model_parameters(model, dec_res)
#         y_pred_proba = model.predict_proba(X_test)[:, 1]
#         y_pred = model.predict(X_test)
#         loss = log_loss(y_test, y_pred_proba)
#         accuracy = accuracy_score(y_test, y_pred)
#         roc_auc = roc_auc_score(y_test, y_pred_proba)
#         f1 = f1_score(y_test, y_pred)
#         print(f'accuracy: {accuracy}')
#         print(f'ROC_AUC: {roc_auc}')
#         print(f'f1_score: {f1}')
        
#         return loss, len(X_test), {"accuracy": accuracy, "roc_auc": roc_auc, "f1-score": f1}

# if __name__ == "__main__":
#     N_CLIENTS = 2
#     parser = argparse.ArgumentParser(description="Flower")
#     parser.add_argument(
#         "--partition-id",
#         type=int,
#         choices=range(0, N_CLIENTS),
#         required=True,
#         help="Specifies the artificial data partition",
#     )
#     args = parser.parse_args()
#     partition_id = args.partition_id
    
#     dataset_train = pd.read_csv(f'./IID_df_{partition_id+1}.csv')
    
#     dataset_test = pd.read_csv(f'./test_glob.csv')

    

#     X_train, y_train = dataset_train.drop(columns=['Fraud']), dataset_train['Fraud']
#     X_test, y_test = dataset_test.drop(columns='Fraud'), dataset_test['Fraud']


#     model = LogisticRegression(
#         penalty='l2',
#         max_iter=1
#     )

#     ts.set_initial_parameters(model)

#     fl.client.start_client(
#         server_address="127.0.0.1:8080",
#         client=CustomClient()
#     )


import flwr as fl
import pickle
import pandas as pd 
import torch
import torch.nn as nn
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, fit_intercept=True):
        super(LogisticRegressionModel, self).__init__()
        self.fit_intercept = fit_intercept
        self.fc = nn.Linear(input_dim, 1, bias=fit_intercept)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

    @property
    def coef_(self):
        return self.fc.weight.detach().numpy()

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self.fc.bias.detach().numpy()
        else:
            return None

    def set_parameters(self, params):
        with torch.no_grad():
            self.fc.weight = nn.Parameter(torch.tensor(params[0], dtype=torch.float32))
            if self.fit_intercept:
                self.fc.bias = nn.Parameter(torch.tensor(params[1], dtype=torch.float32))

    def get_parameters(self):
        if self.fit_intercept:
            return [self.fc.weight.detach().numpy(), self.fc.bias.detach().numpy()]
        else:
            return [self.fc.weight.detach().numpy()]


class CustomClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader, public_key, private_key):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.public_key = public_key
        self.private_key = private_key

    def get_parameters(self, config):
        """Получаем текущие параметры модели."""
        params = self.model.get_parameters()
        print("Текущие параметры:", params)
        return params

    def fit(self, parameters, config):
        """Обучение модели с новыми параметрами."""
        self.model.set_parameters(parameters)
        print("Параметры до обучения:", parameters)

     
        self.train_model(self.model, self.train_loader, self.val_loader, num_epochs=10)


        updated_params = self.model.get_parameters()
        print("Параметры после обучения:", updated_params)


        # encrypted_params = self.encrypt_parameters(updated_params)

        return updated_params, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Оценка модели с параметрами, полученными от сервера."""
        print("Параметры для оценки:", parameters)

        # decrypted_params = self.decrypt_parameters(parameters)
        self.model.set_parameters(parameters)

 
        return self.evaluate_model(self.model, self.test_loader)

    # def encrypt_parameters(self, parameters):
     
    #     encrypted_params = [(self.public_key.encrypt(value),) for param in parameters for value in param.flatten()]
    #     enc_result = []
    #     index = 0

    #     for param in parameters:
    #         num_elements = param.size
    #         reshaped_array = np.array(encrypted_params[index:index + num_elements]).reshape(param.shape)
    #         enc_result.append(reshaped_array)
    #         index += num_elements

    #     return enc_result

    # def decrypt_parameters(self, parameters):
        
    #     decrypted_params = [(self.private_key.decrypt(value),) for param in parameters for value in param.flatten()]
    #     dec_res = []
    #     index = 0

    #     for param in parameters:
    #         num_elements = param.size
    #         reshaped_array = np.array(decrypted_params[index:index + num_elements]).reshape(param.shape)
    #         dec_res.append(reshaped_array)
    #         index += num_elements

    #     return dec_res

    def train_model(self, model, train_loader, val_loader, num_epochs):
  
        criterion = nn.BCELoss()  # Используем BCELoss, так как выход сигмоиды
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_val_roc_auc = 0
        epochs_no_improve = 0
        early_stop_patience = 2  # Количество эпох без улучшений до остановки

        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

            # Валидация после каждой эпохи
            model.eval()
            val_loss, val_roc_auc = 0, 0
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels.float())
                    val_loss += loss.item()
                    predictions = torch.round(outputs)
                    all_labels.extend(labels.numpy())
                    all_predictions.extend(outputs.numpy())

            val_loss /= len(val_loader)
            val_roc_auc = roc_auc_score(all_labels, all_predictions)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val ROC AUC: {val_roc_auc:.4f}')

      
            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                epochs_no_improve = 0  
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print("Ранняя остановка: ROC AUC не улучшился в течение 2 эпох")
                    break


    def evaluate_model(self, model, test_loader):
        """Оценка модели на тестовых данных."""
        model.eval()
        test_loss = 0
        test_labels = []
        test_predictions = []

        criterion = nn.BCELoss()

        with torch.no_grad():
            for test_inputs, test_labels_batch in test_loader:
                test_outputs = model(test_inputs).squeeze()
                loss = criterion(test_outputs, test_labels_batch.float())
                test_loss += loss.item()

                test_labels.extend(test_labels_batch.numpy())
                test_predictions.extend(test_outputs.numpy())

        test_loss /= len(test_loader)
        test_roc_auc = roc_auc_score(test_labels, test_predictions)
        test_f1 = f1_score(test_labels, np.array(test_predictions) > 0.5)
        test_accuracy = accuracy_score(test_labels, np.array(test_predictions) > 0.5)

        print(f'Test Loss: {test_loss:.4f}, Test ROC AUC: {test_roc_auc:.4f}, Test F1 Score: {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        return test_loss, len(test_loader.dataset), {"accuracy": test_accuracy, "roc_auc": test_roc_auc, "f1-score": test_f1}


# Запуск клиента
def load_data(partition_id):
    """Загрузка и подготовка данных для клиента."""
    dataset_train = pd.read_csv(f'./IID_df_{partition_id+1}.csv')
    dataset_test = pd.read_csv(f'./test_glob.csv')

    scaler = MinMaxScaler()
    smote = SMOTE(random_state=22)

    X_train, X_val, y_train, y_val = train_test_split(
        dataset_train.drop(columns='Fraud'),
        dataset_train['Fraud'],
        test_size=0.2,
        random_state=22
    )

    X_train = scaler.fit_transform(X_train)
    X_res_train, y_res_train = smote.fit_resample(X_train, y_train)

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(dataset_test.drop(columns='Fraud'))

    y_test = dataset_test['Fraud']

    train_dataset = TensorDataset(torch.tensor(X_res_train, dtype=torch.float32), torch.tensor(y_res_train.values, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    N_CLIENTS = 2
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the data partition ID"
    )
    args = parser.parse_args()
    partition_id = args.partition_id


    train_loader, val_loader, test_loader = load_data(partition_id)

    model = LogisticRegressionModel(input_dim=train_loader.dataset[0][0].shape[0])

    with open('public_key.pkl', 'rb') as f:
        public_key = pickle.load(f)
    with open('private_key.pkl', 'rb') as f:
        private_key = pickle.load(f)


    client = CustomClient(model, train_loader, val_loader, test_loader, public_key, private_key)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
