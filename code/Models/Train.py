import CNN
import DNN
import RNN
import GRU
import LSTM
import Transformer

def Train(x, y, model = None, epochs = 100, batch_size = 128, LR = 0.001, n_layers = 3, layer_size = 128, filters = 32, dropout = False, MaxPooling = False, Embedding = False, vocab_size = None, embedding_dim = 64, loss = "binary_crossentropy"):
    if model == None:
        print("Please define the model: [CNN,DNN,RNN,GRU,LSTM,Transformer]")
    elif model == "CNN":
        Trained_Model = CNN.Train(x, y, epochs, batch_size, LR, n_layers, filters, dropout, MaxPooling, Embedding, vocab_size, embedding_dim, loss)
    elif model == "DNN":
        Trained_Model = DNN.Train(x, y, epochs, batch_size, LR, n_layers, layer_size, dropout, MaxPooling, Embedding, vocab_size, embedding_dim, loss)
    elif model == "RNN":
        Trained_Model = RNN.Train(x, y, epochs, batch_size, LR, n_layers, layer_size, dropout, MaxPooling, Embedding, vocab_size, embedding_dim, loss)
    elif model == "GRU":
        Trained_Model = GRU.Train(x, y, epochs, batch_size, LR, n_layers, layer_size, dropout, MaxPooling, Embedding, vocab_size, embedding_dim, loss)
    elif model == "LSTM":
        Trained_Model = LSTM.Train(x, y, epochs, batch_size, LR, n_layers, layer_size, dropout, MaxPooling, Embedding, vocab_size, embedding_dim, loss)
    elif model == "Transformer":
        Trained_Model = Transformer.Train(x,y,epochs,batch_size,LR,vocab_size,loss,embedding_dim)
    return Trained_Model

def Eval(x, y, model = None):
    if model == None:
        print("Please provide a trained model.")
        return
    loss, accuracy = model.evaluate(x, y, verbose=False)
    print("Testing Accuracy: {:.4f}".format(accuracy))
