import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
import keras
from pickle import load, dump
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Tools.variableDefinitions import allVarsGen, dijetVars, lumiDict #option to add dipho


def join_objects( X_low_level, low_level_vars_flat,newVars):
    """
        Function take take all low level objects for each event, and transform into a matrix:
           [ [jet1-pt, jet1-eta, ...,
              jet2-pt, jet2-eta, ...,
              jet3-pt, jet3-eta, ... ]_evt1 ,
             [jet1-pt, jet1-eta, ...,
              jet2-pt, jet2-eta, ...,
              jet3-pt, jet3-eta, ...]_evt2 ,
             ...
           ]
        Note that the order of the low level inputs is important, and should be jet objects in descending pT
        Arguments
        ---------
        X_low_level: numpy ndarray
            array of X_features, with columns labelled in order: low-level vars to high-level vars
        Returns
        --------
        numpy ndarray: 2D representation of all jets in each event, for all events in X_low_level
     """
    print 'Creating 2D object vars...'
    l_to_convert = []
    #print 'len X_low_level',len(X_low_level)
    #print 'low_level_vars_flat',len(low_level_vars_flat)
    for index, row in pd.DataFrame(X_low_level, columns= low_level_vars_flat).iterrows(): #very slow; need a better way to do this
        l_event = []
        for i_object_list in newVars:
            l_object = []
            #print 'len newVars',len(newVars)
            for i_var in i_object_list:
                l_object.append(row[i_var])
            l_event.append(l_object)
        l_to_convert.append(l_event)

    print 'Finished creating train object vars'
    return np.array(l_to_convert, np.float32)

def set_model(newVars,dijetVars,n_lstm_layers=3, n_lstm_nodes=150, n_dense_1=1, n_nodes_dense_1=300, n_dense_2=4, n_nodes_dense_2=200, dropout_rate=0.1, learning_rate=0.001, batch_norm=True, batch_momentum=0.99):
    """
        Set hyper parameters of the network, including the general structure, learning rate, and regularisation coefficients.
        Resulting model is set as a class attribute, overwriting existing model.
        Arguments
        ---------
        n_lstm_layers : int
            number of lstm layers/units 
        n_lstm_nodes : int
            number of nodes in each lstm layer/unit
        n_dense_1 : int
            number of dense fully connected layers
        n_dense_nodes_1 : int
            number of nodes in each dense fully connected layer
        n_dense_2 : int
            number of regular fully connected layers
        n_dense_nodes_2 : int
            number of nodes in each regular fully connected layer
        dropout_rate : float
            fraction of weights to be dropped during training, to regularise the network
        learning_rate: float
            learning rate for gradient-descent based loss minimisation
        batch_norm: bool
             option to normalise each batch before training
        batch_momentum : float
             momentum for the gradient descent, evaluated on a given batch
    """
    input_objects = keras.layers.Input(shape=(len(newVars),len(newVars[0])), name='input_objects')
    input_global  = keras.layers.Input(shape=(len(dijetVars),), name='input_global')
    print 'input_objects',input_objects
    print 'input_global', input_global
    lstm = input_objects
    decay = 0.2
    for i_layer in range(n_lstm_layers):
        #lstm = keras.layers.LSTM(n_lstm_nodes, activation='tanh', kernel_regularizer=keras.regularizers.l2(decay), recurrent_regularizer=keras.regularizers.l2(decay), bias_regularizer=keras.regularizers.l2(decay), return_sequences=(i_layer!=(n_lstm_layers-1)), name='lstm_{}'.format(i_layer))(lstm)
        lstm = keras.layers.LSTM(n_lstm_nodes, activation='tanh', return_sequences=(i_layer!=(n_lstm_layers-1)), name='lstm_{}'.format(i_layer))(lstm)
       
 #inputs to dense layers are output of lstm and global-event variables. Also batch norm the FC layers
    dense = keras.layers.concatenate([input_global, lstm])
    print 'dense',dense
    print 'dense_shape',dense.shape
    for i in range(n_dense_1):
        dense = keras.layers.Dense(n_nodes_dense_1, activation='relu', kernel_initializer='he_uniform', name = 'dense1_%d' % i)(dense)
        if batch_norm:
            dense = keras.layers.BatchNormalization(name = 'dense_batch_norm1_%d' % i)(dense)
    dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout1_%d' % i)(dense)

    for i in range(n_dense_2):
        dense = keras.layers.Dense(n_nodes_dense_2, activation='relu', kernel_initializer='he_uniform', name = 'dense2_%d' % i)(dense)
        #add droput and norm if not on last layer
        if batch_norm and i < (n_dense_2 - 1):
            dense = keras.layers.BatchNormalization(name = 'dense_batch_norm2_%d' % i)(dense) 
        if i < (n_dense_2 - 1):
            dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout2_%d' % i)(dense)

    output = keras.layers.Dense(3,activation = 'sigmoid', name = 'output')(dense)
    optimiser = keras.optimizers.Adam(lr = learning_rate)
    
    model = keras.models.Model(inputs = [input_global, input_objects], outputs = [output])
    model.compile(optimizer = optimiser, loss = 'binary_crossentropy')
    return model    

    
