from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pickle
import pandas as pd
import sys
import keras
import keras.backend as K


def RE(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def PR(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def F1(y_true, y_pred):
    precision = PR(y_true, y_pred)
    recall = RE(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def Train(x, y, epochs = 100, batch_size = 128, LR = 0.001, n_layers = 3, layer_size = 128, dropout = False, MaxPooling = False, Embedding = False, vocab_size = None, embedding_dim = 64, loss = "binary_crossentropy",train=False):

    n_layers=2
    embedding_dim=32
    sequence_length=100
    LR=0.0001
    epochs=500
    print("vocab:{} embedding_dim: {} n_layers: {} LR: {}".format(vocab_size,embedding_dim, n_layers, LR))

    model = keras.models.Sequential()
    if Embedding == True:
        model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length=len(x[0])))


    new_shape= (sequence_length,embedding_dim)
    model.add(keras.layers.Reshape((int(sequence_length/5),int(embedding_dim*5)), input_shape=new_shape)) #F1:69
    model.add(keras.layers.GRU(128,unroll=True,dropout=0.3,return_sequences=True))

    for i in range(n_layers):
        if i != n_layers-1:
            model.add(keras.layers.GRU(64,dropout=0.3,unroll=True,return_sequences=True))
        else :
            model.add(keras.layers.GRU(64, dropout=0.3, unroll=False))

    # if dropout == True:
    #     model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.RMSprop(lr=LR), loss='binary_crossentropy', metrics=['accuracy',F1,PR, RE])
    if train == True :
        model.fit(x, y, epochs=epochs,batch_size=batch_size)
    else :
        # model.build(x.shape)
        model.fit(x[:1], y[:1], epochs=1,batch_size=batch_size)
    return model
