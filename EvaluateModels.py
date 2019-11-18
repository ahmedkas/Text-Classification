import numpy as np
import pickle
import pandas as pd
import sys
from keras.models import model_from_json
sys.path.append('./code/DataCleaning')
sys.path.append('./code/Models')
sys.path.append('./code/Evaluation')
from Preprocessing import Preprocess
from DataManipulation import Manipulations_Selector
import Train
import DataSeperator
data = pd.read_csv("/home/sultan/Desktop/bitbuket/Text-Classification-master/data/GT.csv")

x = data["comment_text"]
x = x.values.tolist()
x = Preprocess(x)

available_text_manipulation=sys.argv[1]
available_class=sys.argv[2]
available_model=sys.argv[3]

Embed = False
reshape = True
if available_text_manipulation == "Embedding":
    Embed = True
    reshape = False

if available_text_manipulation == "WORD2VEC":
    reshape = False

if available_text_manipulation == "WORD2VEC_pre":
    print("No reshape")
    reshape = False

if available_text_manipulation == "Glove":
    print("No reshape")
    reshape = False

# Data_Representation, vocab_size = Manipulations_Selector(x,available_text_manipulation,vec_size=100,size=300)
Data_Representation, vocab_size = Manipulations_Selector(x,available_text_manipulation,vec_size=100,size=100)
y = data[available_class]
y = y.values.tolist()
x_train, y_train, x_test, y_test = DataSeperator.Seperate(Data_Representation,y,0.8,reshape=reshape)
if not(available_model == "Transformer" and available_text_manipulation != "Embedding"):

    #### Restore Model ####
    # model = Train.Train(x_train,y_train,model=available_model, epochs = 20, batch_size = 256, LR = 0.001, n_layers = 3, layer_size = 128, filters = 64, dropout = True, MaxPooling = True, Embedding = Embed, vocab_size = vocab_size, embedding_dim = 300, loss = "binary_crossentropy",train=False)
    # model = Train.Train(x_train,y_train, model=available_model, epochs = 30,
    #                     batch_size = 64, LR = 0.001, n_layers = 2,
    #                     layer_size = 128, filters = 64, dropout = True,
    #                      MaxPooling = True, Embedding = Embed,
    #                      vocab_size = vocab_size, embedding_dim = 128,
    model = Train.Train(x_train,y_train, model=available_model, epochs = 30,
                        batch_size = 64, LR = 0.001, n_layers = 3,
                        layer_size = 128, filters = 64, dropout = True,
                         MaxPooling = True, Embedding = Embed,
                         vocab_size = vocab_size, embedding_dim = 128,
                          loss = "binary_crossentropy",train=False)


    model.summary()
    # load weights into new model
    model.load_weights("./models/"+available_text_manipulation+"/"+available_text_manipulation+"-"+available_class+"-"+available_model+".h5")
    print("Loaded model from disk")
    preds = model.predict(x_test)
    f = open("./Predicted/"+available_text_manipulation+"-"+available_class+"-"+available_model+".csv","w")
    text = ""
    for i in range(len(y_test)):
        text += str(y_test[i])+";"+str(preds[i][0])+"\n"
    f.write(text)
