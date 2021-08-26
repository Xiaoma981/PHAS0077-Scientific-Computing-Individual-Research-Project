#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import os




def DataLoader(dataset, answerset):
    infile = open("data/{}".format(str(dataset)), "rb")
    data = pickle.load(infile)
    infile.close()
    answer = np.genfromtxt('data/{}'.format(answerset), usecols=1, dtype=str)
    return data, answer



# two models
def DefineModel_GB():
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='deviance')
    return model



def DefineModel_NN():
    model_nn = MLPClassifier(max_iter=500, shuffle=True)
    return model_nn



def PrepareDataset1(data_size):

    x = []
    y = []

    for k in range(0, 1):
        x_ex, y_ex = DataLoader("data/data_set_{:03d}.bin".format(k), "data/cheatsheet_set_{:03d}.dat".format(k))
        for l in x_ex:
            x.append(x_ex[l][:int(len(x_ex[l])*data_size)])
        for a in y_ex:
            #print(x_ex[l])
            #print(len(x_ex[l])*data_size)
            template = np.empty((int(len(x_ex[l])*data_size))).astype(str)
            template[:] = a
            y.append(template)

    del x_ex
    del y_ex
    del template

    x = np.vstack((np.array(x)))
    y = np.hstack((np.array(y)))

    return x, y



def PrepareDataset(beggining, end):

    x = []
    y = []

    for d_num in range(beggining, end):
        x_ex, y_ex = DataLoader("data/data_set_{:03d}.bin".format(d_num), "data/cheatsheet_set_{:03d}.dat".format(d_num))
        for simulation, answer in zip(x_ex, y_ex):
            x.append(x_ex[simulation])
            template = np.empty(len(x_ex[simulation])).astype(str)
            template[:] = answer
            y.append(template)

    del x_ex
    del y_ex
    del template

    x = np.vstack((np.array(x)))
    y = np.hstack((np.array(y)))

    return x, y



def Train_method(x, y, model, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    trained_model = model.fit(x_train, y_train)
    y_test_pred = trained_model.predict(x_test)
    y_train_pred = trained_model.predict(x_train)

    #Plotting the Confusion Matrices
    conf_matrix(trained_model, x_test, y_test, "Test")
    conf_matrix(trained_model, x_train, y_train, "Train")

    return trained_model, x_train, y_test_pred, y_test, y_train_pred, y_train


def SaveModel(trained_model, flag):

    file = open("trained_model_{}.nn".format(flag), 'wb')
    pickle.dump(trained_model, file)
    file.close()

    return


def save_score(number,result):
    
    path_file_name = 'result/result_{}.txt'.format(number)
    if not os.path.exists(path_file_name):
        with open(path_file_name, "w") as f:
            print(f)

    with open(path_file_name, "a") as f:
        f.write(result)



def show_confusion_matrix(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred, ["IN", "OUT"])
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices)
    plt.yticks(indices)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])
    plt.show()



def conf_matrix(model, x, y, title):
    """
    Function for prediction + plotting of the model for the given data (x,y)

    :param model: (sklearn model / object) Trained model that is going to be used for the conf matrix.
    :param x: (array) X data to use.
    :param y: (array) Y data to use.
    :param title: (str) Title of the plot.
    :return: empty (shows the plots)
    """
    plot_confusion_matrix(model, x, y,
                          cmap=plt.cm.Blues, normalize="true")
    plt.title(title)
    plt.show()
    return



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    plt.xlim([-7000, 7000])
    plt.show()



def validation(model, n_datasets, fraction):
    overall_acc = []

    for ndat in range(0, n_datasets):
        x, y = PrepareDataset(ndat, ndat+1)
        x = x[int(-len(x)*fraction):]
        y = y[int(-len(y)*fraction):]
        y_pred = model.predict(x)
        results = y == y_pred
        acc = results.mean()*100
        overall_acc.append(acc)

    overall_acc = np.array(overall_acc).mean()
    print("Total accuracy on whole dataset using", fraction*100, " of the end is ", overall_acc, "%")

    return overall_acc



def Hyperparameter_search(option, model, parameter_grid):
    if option == "Random":
        sampler = RandomizedSearchCV(model,parameter_grid, scoring="accuracy")
    elif option == "Exhaustive":
        sampler = GridSearchCV(model, parameter_grid, scoring="accuracy")
    else:
        sampler = False
        print("You did not specify an available option")
    if sampler == False:
        raise Exception("No Sampler specified")

    x, y = PrepareDataset1(0.001)
    sampler.fit(x, y)
    grid_results = sampler.cv_results_
    
    return grid_results



def first_train():
    model = DefineModel_NN()
    x, y = PrepareDataset1(0.1)
    trained_model,x_train, y_test_pred, y_test, y_train_pred, y_train = Train_method(x, y, model, 0.4)
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
    print('Test accuracy score =', test_accuracy*100, '%')
    print('Train accuracy score =', train_accuracy*100, '%')
    show_confusion_matrix(y_test,y_test_pred)



parameter_grid_nn = {"solver": ['sgd','adam','lbfgs'],
                  "activation" :['identity’, ‘logistic’, ‘tanh’, ‘relu'],
                  "max_iter": 10,
                  "alpha": np.linspace(0.000001, 0.001, 10),
                  "hidden_layer_sizes": [(100,50),(150,25)],
                  "random_state": 1,
                  "verbose": True
    }

parameter_grid_gb = {"n_estimators": 100,
                     "learning_rate": 0.1,
                     "max_depth": 1,
                     "random_state": 0,
                     "loss": 'deviance'
    }

    


if __name__ == '__main__':

    first_train()

    #EXAMPLE
    param_grid = {
        'hidden_layer_sizes': [100, 120, 150, 160, 190, 200],
        'activation': ('identity', 'logistic', 'tanh', 'relu'),
        'solver': ('lbfgs', 'sgd', 'adam')
                  }
