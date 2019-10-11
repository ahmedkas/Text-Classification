import numpy as np
import pickle
import pandas as pd
import sys
sys.path.append('./code/DataCleaning')
sys.path.append('./code/Models')
sys.path.append('./code/Evaluation')
from Preprocessing import Preprocess
from DataManipulation import Manipulations_Selector
import Train
import DataSeperator
data = pd.read_csv("./data/GT.csv")

x = data["comment_text"]
x = x.values.tolist()
x = Preprocess(x)

Available_Classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
Available_Models = ['CNN','DNN','RNN','LSTM','GRU','Transformer']
Available_Text_Manipulations = ['DOC2VEC','Embedding','USE']


for available_text_manipulation in Available_Text_Manipulations:
    Embed = False
    reshape = True
    if available_text_manipulation == "Embedding":
        Embed = True
        reshape = False
    Data_Representation, vocab_size = Manipulations_Selector(x,available_text_manipulation,vec_size=64,size=300)
    for available_class in Available_Classes:
        y = data[available_class]
        y = y.values.tolist()
        x_train, y_train, x_test, y_test = DataSeperator.Seperate(Data_Representation,y,0.8,reshape=reshape)
        for available_model in Available_Models:
            if available_model == "Transformer" and available_text_manipulation != "Embedding":
                continue
            model = Train.Train(x_train,y_train,model=available_model, epochs = 20, batch_size = 256, LR = 0.001, n_layers = 1, layer_size = 128, filters = 64, dropout = True, MaxPooling = True, Embedding = Embed, vocab_size = vocab_size, embedding_dim = 300, loss = "binary_crossentropy")

            Train.Eval(x_test,y_test,model)
            model_json = model.to_json()
            with open("./models/"+available_text_manipulation+"-"+available_class+"-"+available_model+".json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("./models/"+available_text_manipulation+"-"+available_class+"-"+available_model+".h5")
            print("Saved model "+available_text_manipulation+"-"+available_class+"-"+available_model+" to disk")
