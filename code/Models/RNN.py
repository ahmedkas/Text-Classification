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
        if i != n_layers-1:
            model.add(keras.layers.SimpleRNN(32,unroll=True,return_sequences=True))
        else :
            model.add(keras.layers.SimpleRNN(32,unroll=False))

    if dropout == True:
        model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.Dense(len(y[0]), activation='softmax'))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(lr=LR), loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x, y, epochs=epochs,batch_size=batch_size)
    return model
