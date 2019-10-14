import numpy as np
import pickle
import pandas as pd
import sys
import keras

def Train(x, y, epochs = 100, batch_size = 128, LR = 0.001, n_layers = 3, layer_size = 128, dropout = False, MaxPooling = False, Embedding = False, vocab_size = None, embedding_dim = 64, loss = "binary_crossentropy"):



    model = keras.models.Sequential()

    if Embedding == True:
        model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length=len(x[0])))
    for i in range(n_layers):
        model.add(keras.layers.Dense(layer_size, activation='relu'))
        if dropout == True:
            model.add(keras.layers.Dropout(0.2))
        if MaxPooling == True:
            model.add(keras.layers.MaxPooling1D(pool_size=3, strides=None))

    model.add(keras.layers.GlobalMaxPooling1D())
    # model.add(keras.layers.Dense(len(y[0]), activation='softmax'))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(lr=LR), loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x, y, epochs=epochs,batch_size=batch_size)
    return model
