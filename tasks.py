import numpy as np
import os 
import json
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def get_model_parameters(model):

    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_,
                  ]

        
    return params



def set_model_parameters(model, params):
    
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_parameters(model):
    n_features = 594 
    model.classes_ = np.arange(1)
    model.coef_ = np.random.randn(1, n_features)  
    if model.fit_intercept:
        model.intercept_ = np.random.randn(1)  




def save_metrics_json(client, strategy_suffix, filename="./met_.json"):
    
    metrics = {
        "losses": list(client.losses),
        "ROC_AUCs": list(client.ROC_AUCs),
        "ACCURACYs": list(client.ACCURACYs),
        "F1s": list(client.F1s)
    }

    
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as f:
            try:
                all_metrics = json.load(f)
            except json.JSONDecodeError:
                
                all_metrics = {}
    else:
        all_metrics = {}

    
    if strategy_suffix not in all_metrics:
        all_metrics[strategy_suffix] = []
    
    all_metrics[strategy_suffix].append(metrics)

   
    os.makedirs(os.path.dirname(filename), exist_ok=True)  
    with open(filename, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Метрики сохранены в {filename} с суффиксом {strategy_suffix}")



def prepare_data(df, X_test):


    scale = MinMaxScaler()
    smote = SMOTE(random_state=22)

    X_, y_ = df.drop(columns="Fraud"), df.Fraud
    X_train, X_val, y_train, y_val = train_test_split(X_, y_ ,test_size=0.2, random_state=222)

    X_train_scale = scale.fit_transform(X_train)
    X_val = scale.transform(X_val)
    X_test = scale.transform(X_test)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scale)

    X_smote, y_smote = smote.fit_resample(X_train_poly, y_train)
    X_val = poly.transform(X_val)
    X_test = poly.transform(X_test)
    return X_smote, X_val, X_test, y_smote, y_val