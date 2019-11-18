import numpy as np
import pickle
import pandas as pd
import sys
import keras
from keras import backend as K
from keras import regularizers



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

def Train(x, y, epochs = 1000, batch_size = 128, LR = 0.001, n_layers = 2,
            filters = 32, dropout = False, MaxPooling = False,
            Embedding = False, vocab_size = None, embedding_dim = 64,
            loss = "binary_crossentropy",train=False):


    model = keras.models.Sequential()

    if Embedding == True:
        print("Vocab size: "+str(vocab_size))
        print("Added Embedding Layer")
        model.add(keras.layers.Embedding(vocab_size, embedding_dim,
        input_length=len(x[0])))

    for i in range(n_layers):
        model.add(keras.layers.Conv1D(16, 8,activation='relu'))

    model.add(keras.layers.Dropout(0.5))
    for i in range(n_layers):
        model.add(keras.layers.Conv1D(64, 4, activation='relu'))

    if MaxPooling == True:
        model.add(keras.layers.MaxPooling1D(pool_size=3, strides=None))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(128, activation='relu'))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(lr=LR), loss='binary_crossentropy', metrics=['acc',F1,PR, RE])
    # model.compile(optimizer=keras.optimizers.Adam(lr=LR), loss='mean_squared_error', metrics=['accuracy'])
    if train == True :
        model.fit(x, y, epochs=epochs,batch_size=batch_size, verbose=1)
    else :
        # Called to build a model for the evaluation
        model.fit(x[:1], y[:1], epochs=1,batch_size=batch_size)
    return model
