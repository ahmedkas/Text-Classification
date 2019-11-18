import os


# Available_Classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
Available_Classes = ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Available_Classes = ['toxic']
# Available_Models = ['CNN','DNN','RNN','LSTM','GRU','Transformer']
Available_Models = ["GRU","LSTM"]
# Available_Text_Manipulation = ['DOC2VEC','Embedding','USE','WORD2VEC','n_grams','WORD2VEC_pre','Glove']
Available_Text_Manipulations = ['WORD2VEC']
for available_text_manipulation in Available_Text_Manipulations:
    for available_class in Available_Classes:
        for available_model in Available_Models:
            cmd = 'python main.py '+available_text_manipulation+" "+available_class+" "+available_model
            os.system(cmd)
