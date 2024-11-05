import numpy as np

def get_model_parameters(model):

    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_]

        
    return params



def set_model_parameters(model, params):
    
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_parameters(model):
    
    n_features = 33 # Number of features in dataset
    model.classes_ = np.array([0, 1])
    model.coef_ = np.zeros((1, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))