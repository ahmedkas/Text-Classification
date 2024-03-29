import numpy as np
import pickle
import pandas as pd
import sys
from collections import defaultdict

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

if available_text_manipulation == "WORD2VEC_pre":
    #print("No reshape")
    reshape = False

if available_text_manipulation == "WORD2VEC":
    #print("No reshape")
    reshape = False

if available_text_manipulation == "Glove":
    #print("No reshape")
    reshape = False



Data_Representation, vocab_size = Manipulations_Selector(x,available_text_manipulation,
                                        vec_size=100,size=100,features_words=1000,
                                        features_chars=5000,word_min = 1,
                                        word_max=3,char_min = 2,char_max = 5)
#print("in main, vocab_siz: ="+str(vocab_size))
y = data[available_class]
y = y.values.tolist()
x_train, y_train, x_test, y_test = DataSeperator.Seperate(Data_Representation,y,0.8,reshape=reshape)

# def getClassWeights(data):
#     dict_= defaultdict(int)
#     for i in data:
#         dict_[i]+=1
#     return dict_
# class_weight=getClassWeights(y_train)



if not(available_model == "Transformer" and available_text_manipulation != "Embedding"):
    # model = Train.Train(x_train,y_train, model=available_model, epochs = 30,
    #                     batch_size = 64, LR = 0.001, n_layers = 2,
    #                     layer_size = 128, filters = 64, dropout = True,
    #                      MaxPooling = True, Embedding = Embed,
    #                      vocab_size = vocab_size, embedding_dim = 128,
    #                       loss = "binary_crossentropy",train=True, class_weight=class_weight)
    model = Train.Train(x_train,y_train, model=available_model, epochs = 30,
                        batch_size = 128, LR = 0.001, n_layers = 2,
                        layer_size = 128, filters = 64, dropout = True,
                         MaxPooling = True, Embedding = Embed,
                         vocab_size = vocab_size, embedding_dim = 128,
                          loss = "binary_crossentropy",train=True)


    Train.Eval(x_test,y_test,model)
    model_json = model.to_json()
    with open("./models/"+available_text_manipulation+"-"+available_class+"-"+available_model+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./models/"+available_text_manipulation+"-"+available_class+"-"+available_model+".h5")
    print("Saved model "+available_text_manipulation+"-"+available_class+"-"+available_model+" to disk")
