#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time
import optuna
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from dataManager import DataSet
import dataManager as dm




def save_result(study,studyname):
    infile = open("{}.pkl".format(studyname), "wb")
    pickle.dump(study, infile)
  
    
  
def load_result(studyname):
    infile = open("{}.pkl".format(studyname), "rb")
    studyname = pickle.load(infile)
    return studyname



def tf_train(x_train, y_train, x_test, y_test, training_model_trial, 
             extra_info, tol_function=1e-4, max_patience_function =10
             ,batch_size_function=20,hidden_nodes_function=100,
             optimizer_function="adam",learning_rate_function=0.001,
             beta_1_function=0.9,beta_2_function=0.999,epsilon_function=1e-08,
             activation_function='relu', loss_function='log_loss', 
             rho_function=0.95,initial_accumulator_value_function=0.1,
             n_epoch_function=500,hidden_layer_function=1):
    """
    :param x_train: feature for training
    :param y_train: label for training
    :param x_test:  feature for test
    :param y_test: label for testing
    :param training_model_trial: the ordinal number of this model
    :param extra_info: dict
    :return: accuracy_train:accuracy on training set of each epoch
             accuracy_test: accuracy on testing set of each epoch
             loss_list: loss of each epoch
    """

    tolerance =  tol_function
    max_patience = max_patience_function
    n_epoch = n_epoch_function
    layer_number = hidden_layer_function
    layer_size = hidden_nodes_function
    feature_number=len(x_train[0])
    train_set = DataSet(x_train, y_train)  # split data into batches
    batch_size = batch_size_function
    hidden_nodes = hidden_nodes_function
    optimizer = optimizer_function
     
    
    n_batch = train_set.num_examples // batch_size
    best_loss = sys.maxsize
    number_of_patience = 0
    accuracy_train = []
    accuracy_test = []
    loss_list = []
    graph = tf.Graph()  # initialize graph before construct it otherwise saver will save empty graph in a loop
    
    with graph.as_default():
        x = tf.compat.v1.placeholder(tf.float32, [None, feature_number], name='x')  # Input layer
        y = tf.compat.v1.placeholder(tf.float32, [None, 2], name='y')  # Output Layer
        # first layer
        # construct layers
        layers=[]
        layers.append(x)  # add initial layer
        last_layer_size=feature_number
        for i in range(layer_number):
            weight_name="weight_l{}".format(i+1)  # the name of weight tensor
            weights_li = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([last_layer_size, layer_size], stddev=0.1), name=weight_name)
            bias_name="bias_l{}".format(i+1)  # the name of bias tensor
            b_li = tf.Variable(tf.zeros([layer_size]) + 0.01, name=bias_name)
            li = tf.matmul(layers[-1], weights_li) + b_li
            # chose activation function
            if activation_function =='relu':
                li = tf.nn.relu(li)
            elif activation_function == 'elu':
                li = tf.nn.elu(li)
            elif activation_function =='gelu':
                li = tf.nn.gelu(li)
            elif activation_function =='selu':
                li = tf.nn.selu(li)
            layers.append(li)
            last_layer_size=layer_size

            
        weights_lout = tf.Variable(tf.compat.v1.truncated_normal([last_layer_size, 2], stddev=0.1), name='weight_lout')
        b_lout = tf.Variable(tf.zeros([2]) + 0.01, name='biases_lout')
        output = tf.matmul(layers[-1], weights_lout) + b_lout
        prediction = tf.nn.softmax(output, name='prediction')

        # chose loss function
        if loss_function =='log_loss':
            loss = tf.compat.v1.losses.log_loss(labels=y, predictions=prediction)
        elif loss_function =='huber_loss':
            loss = tf.compat.v1.losses.huber_loss(labels=y, predictions=prediction)
        elif loss_function =='mean_squared_error':
            loss = tf.compat.v1.losses.mean_squared_error(labels=y, predictions=prediction)
        elif loss_function =='mean_pairwise_squared_error':
            loss = tf.compat.v1.losses.mean_pairwise_squared_error(labels=y, predictions=prediction)
            
        # chose optimizer
        if optimizer == "adam":
            learning_rate = learning_rate_function
            beta_1 = beta_1_function
            beta_2 = beta_2_function
            epsilon = epsilon_function
            train = tf.compat.v1.train.AdamOptimizer(learning_rate,beta_1,beta_2,epsilon).minimize(loss)
        elif optimizer == "adadelta":
            learning_rate = learning_rate_function
            rho = rho_function
            epsilon = epsilon_function
            train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate,rho,epsilon).minimize(loss)
        elif optimizer == "adagrad":
            learning_rate = learning_rate_function
            initial_accumulator_value = initial_accumulator_value_function
            train = tf.compat.v1.train.AdagradOptimizer(learning_rate,initial_accumulator_value).minimize(loss)
        elif optimizer == "GradientDescent":
            learning_rate = learning_rate_function
            train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        elif optimizer == "momentum":
            learning_rate = learning_rate_function
            train = tf.compat.v1.train.MomentumOptimizer(learning_rate,0.1).minimize(loss)
            
            
           
        correct_prediction = tf.equal(tf.compat.v1.arg_max(y, 1), tf.compat.v1.arg_max(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(n_epoch):
            for batch in range(n_batch):
                x_batch, y_batch = train_set.next_batch(batch_size)
                sess.run(train, feed_dict={x: x_batch, y: y_batch})
            acc_test = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
            acc_train = sess.run(accuracy, feed_dict={x: x_train, y: y_train})
            accuracy_test.append(acc_test)
            accuracy_train.append(acc_train)
            current_loss = sess.run(loss, feed_dict={x: x_train, y: y_train})
            loss_list.append(current_loss)
            # early stop procedure
            if current_loss > best_loss - tolerance:
                number_of_patience = number_of_patience + 1
            else:
                number_of_patience = 0
            if current_loss < best_loss:
                best_loss = current_loss
            if number_of_patience >= max_patience:
                break

    return accuracy_train, accuracy_test, loss_list

    

def objective(trial):
    args = sys.argv
    n_epoch_function = trial.suggest_int("n_epoch",100,10000,log = True)
    tol_function = trial.suggest_float("tol",1e-4, 0.1, log = True)
    max_patience_function = trial.suggest_int("max_patience",1,50,log = True)
    batch_size_function  = trial.suggest_int("batch_size", 1000, 2500, log=True)
    hidden_nodes_function = trial.suggest_int("hidden_nodes", 50, 500, log=True)
    optimizer_function = trial.suggest_categorical("optimizer",['adam','adadelta','adagrad','GradientDescent','momentum'])
    learning_rate_function = trial.suggest_float("learning_rate", 0.0001, 0.1, log=True)
    beta_1_function = trial.suggest_float("beta_1", 0.1, 0.9, log = True)
    beta_2_function = trial.suggest_float("beta_2", 0.001, 0.999, log = True)
    epsilon_function = trial.suggest_float("epsilon", 1e-9, 1e-2, log = True)
    rho_function = trial.suggest_float("rho", 0.1, 0.999, log = True)
    initial_accumulator_value_function = trial.suggest_float("initial_accumulator_value", 0.1, 0.9, log = True)
    activation_function = trial.suggest_categorical("activation", ["elu", "gelu", "selu", "relu"])
    loss_function = trial.suggest_categorical("loss", ["log_loss", "huber_loss", "mean_squared_error", "mean_pairwise_squared_error"])
    hidden_layer_function = trial.suggest_int("hidden_layer",1,30,log = False)                        
                                              
    
    data_path = "data/"
    subset = 0
    ds=0.3/5
    print('data size is ', ds)
    x, y = dm.get_data_integrated(data_path, subset, ds, 0, 1, latter_cut_percentage=0.1, opt='equilibrium', data_set_version='old')
    y = np.reshape(y, (len(y), 1))
    encoder = OneHotEncoder()
    encoder.fit(y)
    y = encoder.transform(y).toarray()
    print(x.shape[1], " features in this dataset")
    extra_info = {}
    extra_info["input_nodes"] = x.shape[1]
    models_directory = 'models_subset_{:03d}/'.format(subset)
    extra_info['directory'] = models_directory
    if not os.path.exists(models_directory):
        os.mkdir(models_directory)
    best_accuracy = 0
    all_accuracy = []
    
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        accuracy_train, accuracy_test, loss_list = tf_train(x_train, 
                        y_train, x_test, y_test, i, extra_info,
                        tol_function,max_patience_function, 
                        batch_size_function,hidden_nodes_function, 
                        optimizer_function,learning_rate_function,
                        beta_1_function,beta_2_function,epsilon_function,
                        activation_function,loss_function,rho_function,
                        initial_accumulator_value_function,n_epoch_function,hidden_layer_function)
        all_accuracy.append(accuracy_test[-1])
        if (accuracy_test[-1] > best_accuracy):
            best_accuracy = accuracy_test[-1]
            
    std = np.std(all_accuracy)
    print("std is: ",std)
    print("best accuarcy is ",best_accuracy)
    
    return best_accuracy,std
    
    
studyname = "MLPC_Opt"
study = optuna.create_study(directions = ['maximize','minimize'], study_name=studyname)
study.optimize(objective, n_trials = 200)
save_result(study,studyname)







