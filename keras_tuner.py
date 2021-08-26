#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np 
import kerastuner as kt
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier




def DataLoader(dataset, answerset):
    infile = open("{}".format(str(dataset)), "rb")
    data = pickle.load(infile)
    infile.close()
    answer = np.genfromtxt('{}'.format(answerset), usecols=1, dtype=str)
    return data, answer


def PrepareDataset(data_size, n_datasets_range, percentage):

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

""" First we state the build model which is the hypermodel we will use for our hyperparameter search"""
def build_model(hp):
    
    """ This is a made up argument from kerastuner to make it choose between options if you wish
    you can make your custom choices here which are not implemented in the sklearn model yet, for example
    you could even change the model itself depending on these, for example I will implement Gradient Boosting and 
    the neural network we used."""
    
    model_type = hp.Choice('model_type', ['gradient_boosting','neural_network']) 
    
    
    """ Now we put the conditionals so we can change model as well """
    if model_type == 'gradient_boosting':
        # Here we call the model itself from sklearn
        model = ensemble.GradientBoostingClassifier(
        # Now we call all arguments we want to be able to alter
        # This is what we do with string choices
        loss=hp.Choice("loss", ["deviance"]),
        # This is what we do when we want to alter a float, we can even set the way it samples from it
        learning_rate=hp.Float("learning_rate", 1e-3, 1, sampling="log"),
        # This is for selecting integers, we can also set the step at which it will try
        n_estimators=hp.Int('n_estimators', 5, 250, step=10),
        subsample = hp.Float("subsample", 0.2, 1, sampling='log'),
        criterion = hp.Choice('criterion',['friedman_mse', 'mse']),
        min_samples_split = hp.Float('min_samples_split',0.1,0.9,step =0.1),
        min_samples_leaf = hp.Float('min_samples_leaf',0.1,0.9,step =0.1),
        min_weight_fraction_leaf = hp.Float('min_weight_fraction_leaf',0.1,0.4,step =0.01),
        max_depth = hp.Int('max_depth',3,10,step =1),
        min_impurity_decrease = hp.Float('min_impurity_decrease',0.0,0.9,step =0.1),
        verbose = hp.Int('verbose',0,10,step =1),
        max_leaf_nodes = hp.Int('max_leaf_nodes',5,20,step =1),
        warm_start = ('warm_start',False),
        validation_fraction = hp.Float('validation_fraction',0.01,0.75,step =0.01),
        n_iter_no_change = hp.Int('n_iter_no_change',1,10,step =1),
        tol = hp.Float('tol',1e-5,1e-3,sampling ='log'),
        )
        
        
    elif model_type == 'neural_network':
        model = MLPClassifier(
        activation = hp.Choice("activation", ["identity", 'logistic', 'tanh', 'relu']),
        solver = hp.Choice("solver", ['sgd', 'adam']), #Attention "lbfgs" does not work
        alpha=hp.Float('alpha', 1e-3, 1, sampling='log'),
        batch_size = hp.Int("batch_size", 10, 20, step=1),
        max_iter = hp.Int("max_iter", 200, 800, step=25),
        shuffle = hp.Boolean('shuffle',False),
        tol = hp.Float('tol',1e-5,1e-3,sampling ='log'),
        verbose = hp.Boolean('verbose',False),
        warm_start = hp.Boolean('warm_start',False),
        early_stopping = hp.Boolean('early_stopping',False),
        n_iter_no_change = hp.Int('n_iter_no_change',10,100,step =5)    
        )
    return model


tuner=kt.tuners.Sklearn(
    #Here we specify the oracle we want to use (the way of optimizing the parameters)
    oracle=kt.oracles.BayesianOptimization(
    #oracle=kt.oracles.RandomSearch(
        #Here we specify our goal with this parameter search
        objective=kt.Objective('score', 'max'),
        #Maximum number of trials
        max_trials=100),
    #Specify the model to use (the previous one)
    hypermodel=build_model,
    #How to set the scoring method
    scoring=metrics.make_scorer(metrics.accuracy_score),
    #What is going to be our CV of choice? 
    cv=model_selection.StratifiedKFold(5),
    #cv=ShuffleSplit(n_splits=3, test_size=0.3, random_state=10),
    #Directory where to create our folder and put the results in that folder
    directory='.',
    #The name of our project folder to save the models
    project_name='test_nn_real_data')



""" Our data """
n_datasets= range(1,0)
X, y = PrepareDataset(0.06, n_datasets, 0.10)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3)

tuner.search(X_train, y_train)

best_model = tuner.get_best_models(num_models=1)[0]
print(best_model)


