#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
from sklearn.neural_network import MLPClassifier
import time
import optuna




def DataLoader(dataset, answerset):
    infile = open("{}".format(str(dataset)), "rb")
    data = pickle.load(infile)
    infile.close()
    answer = np.genfromtxt('{}'.format(answerset), usecols=1, dtype=str)
    return data, answer


def PrepareDatasetSp(data_size, n_datasets_range, percentage):

    x = []
    y = []

    for k in n_datasets_range:
        x_ex, y_ex = DataLoader("data/data_set_{:03d}.bin".format(k), "data/cheatsheet_set_{:03d}.dat".format(k))
        for l in x_ex:
            sim = x_ex[l][:int(len(x_ex[l])*data_size)]
            x.append(sim[int(abs(len(np.array(sim))*percentage - len(np.array(sim)))):])
        for a in y_ex:
            template = np.empty((int(len(x_ex[l])*data_size*percentage))).astype(str)
            template[:] = a
            y.append(template)
        del x_ex
        del y_ex
    
    del template

    x = np.vstack((np.array(x)))
    y = np.hstack((np.array(y)))

    return x, y

"""
def showgraph():
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[2, 1, 4, 3]))
    fig.add_trace(go.Bar(y=[1, 4, 3, 2]))
    fig.update_layout(title = 'Hello Figure')
    fig.show()
"""

def MLP_Classifier(trial):
    
    chk = time.time()
    
    """ Model Bulding """
    
    #Here we describe the parameters we wish to sample
    activation_function = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
    solver_function = trial.suggest_categorical("solver", ["sgd", "adam"])
    batch_size_function  = trial.suggest_int("batch_size", 500, 5000, log=True)
    alpha_function = trial.suggest_float("alpha", 0.00001, 0.9, log=True)
    learning_rate_function = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
    learning_rate_init_function = trial.suggest_float("learning_rate_init", 1e-8, 1e-3, log=True)
    power_t_function = trial.suggest_float("power_t", 0.01, 0.9, log=True)
    max_iter_function = trial.suggest_int("max_iter", 10, 500, log = False)
    shuffle_function = trial.suggest_categorical("shuffle", [True])
    random_state_function = 42
    #trial.suggest_int("random_state", 1,100,log = False)
    tol_function = trial.suggest_float("tol",1e-4, 0.1, log = True)
    verbose_function = trial.suggest_categorical("verbose", [False])
    warm_start_function = trial.suggest_categorical("warm_start", [True, False])
    momentum_function = trial.suggest_float("momentum", 0.01, 0.9, log = True)
    nesterovs_momentum_function = trial.suggest_categorical("nesterovs_momentum", [True, False])
    early_stopping_function = trial.suggest_categorical("early_stopping", [True, False])
    validation_fraction_function = trial.suggest_float("validation_fraction", 0.01, 0.9, log = True)
    beta_1_function = trial.suggest_float("beta_1", 0.01, 0.9, log = True)
    beta_2_function = trial.suggest_float("beta_2", 0.01, 0.9, log = True)
    epsilon_function = trial.suggest_float("epsilon", 1e-8, 1e-4, log = True)
    n_iter_no_change_function = trial.suggest_int("n_iter_no_change", 2, 20, log = False)
    max_fun_function = trial.suggest_int("max_fun", 1000, 20000, log = False)
    
 
    #pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    classifier_obj = MLPClassifier(
        #max_iter=500,
        activation = activation_function, 
        solver=solver_function, 
        batch_size = batch_size_function, 
        alpha = alpha_function,
        learning_rate = learning_rate_function,
        learning_rate_init = learning_rate_init_function,
        power_t = power_t_function,
        max_iter = max_iter_function,
        shuffle = shuffle_function,
        random_state = random_state_function,
        tol = tol_function,
        verbose = verbose_function,
        warm_start = warm_start_function,
        momentum = momentum_function,
        nesterovs_momentum = nesterovs_momentum_function,
        early_stopping = early_stopping_function,
        validation_fraction = validation_fraction_function,
        beta_1 = beta_1_function,
        beta_2 = beta_2_function,
        epsilon = epsilon_function,
        n_iter_no_change = n_iter_no_change_function,
        max_fun = max_fun_function
        
        )
    
    """ Fitting the model """
    
    #We load the datasets for training first
    dsets = range(0,1)
    x, y = PrepareDatasetSp(0.3/5, dsets, 0.1)
    
    #After having stated them, we do a fitting with a cross validation score
    cv_score = sklearn.model_selection.cross_val_score(
        classifier_obj, x, y, n_jobs=-1, cv=5)
    cv_acc = cv_score.mean()
    print("CV Accuracy:", cv_acc)
    print("Training with all data")
    
    classifier_obj.fit(x, y)
    test_accuracy = classifier_obj.score(x, y)
    print("Test Accuracy:", test_accuracy)
    
    
    """ Validation part """
    #Now we load the data for the validation of the model
    dsets = range(1,2)
    x, y = PrepareDatasetSp(0.3/5, dsets, 0.1)
    
    #We predict the labels of the new datasets never seen by the model 
    #Now we calculate the score 
    validation_acc = classifier_obj.score(x, y)
    print("Validation Accuracy:", validation_acc)
    print("Run done in", (time.time() - chk)/60, "minutes")
    
    return cv_acc, validation_acc


def save_result(studyname,name):
    infile = open(name, "wb")
    pickle.dump(studyname, infile)
    
def load_result():
    infile = open("study.pkl", "rb")
    studyname = pickle.load(infile)
    return studyname




study = optuna.create_study(
    directions=["maximize", "maximize"],
)
study.optimize(MLP_Classifier, n_trials=200)
save_result(study,"study1.pkl")


"""
study2 = optuna.create_study(
    directions=["maximize", "maximize"],
    pruner=optuna.pruners.PercentilePruner(
        25.0, n_startup_trials=5, n_warmup_steps=30, interval_steps=10
    ),
)
study2.optimize(MLP_Classifier, n_trials=200)
save_result(study2,"study2.pkl")



study3 = optuna.create_study(
    directions=["maximize", "maximize"],
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=10, reduction_factor=3
    ),
)
study3.optimize(MLP_Classifier, n_trials=200)
save_result(study3,"study3.pkl")


study4 = optuna.create_study(
    directions=["maximize", "maximize"],
    pruner=optuna.pruners.MedianPruner(
       n_startup_trials=5, n_warmup_steps=30, interval_steps=10
    ),
)
study4.optimize(MLP_Classifier, n_trials=200)
save_result(study4,"study4.pkl")
"""
    




