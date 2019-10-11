import nltk
import re
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import os.path
import tensorflow_hub as hub
import keras
import pickle
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def Manipulations_Selector(docs,option=None,vec_size=64,size=100):
    vocab_size = None
    if option == None:
        print("Please provide the option [USE,DOC2VEC,Embedding]")
    elif option == "USE":
        docs = USE(docs)
    elif option == "DOC2VEC":
        docs = DOC2VEC(docs,vec_size)
    elif option == "Embedding":
        docs,vocab_size = Embedding(docs,size)
    else :
        print("The provided option not valid [USE,DOC2VEC,Embedding]")
        return None,None
    return docs,vocab_size


def USE(docs):
    vectors = []
    if not os.path.exists('./data/ProcessedData/USEData'):
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        for j in range(int(len(docs)/1000) + 1 ):
            sess = tf.Session()
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            start = j*1000
            end = (j+1)*1000
            embeddings = sess.run(embed(docs[start:end]))
            embeddings = np.array(embeddings).tolist()
            for i in range(len(embeddings)):
                vectors.append(embeddings[i])
            sess.close()

        f = open("./data/ProcessedData/USEData","wb")
        pickle.dump(vectors,f)
    else :
        f = open("./data/ProcessedData/USEData","rb")
        vectors = pickle.load(f)
    return vectors


def Embedding(docs,size = 100):
    vectors = []
    if not os.path.exists('./data/ProcessedData/EmbeddingData'):
        Vocab = []
        indeces = {}
        for i in range(len(docs)):
            docs[i] = docs[i].split(" ")
            docs[i] = docs[i][:size]
        if not os.path.exists('./data/ProcessedData/EmbeddingVocab'):
            for i in range(len(docs)):
                for word in docs[i]:
                    if word != "":
                        Vocab.append(word)
            Vocab = list(set(Vocab))
            indeces = {}
            for i in range(len(Vocab)):
                indeces[Vocab[i]] = i
            f = open("./data/ProcessedData/EmbeddingVocab","wb")
            pickle.dump(Vocab,f)
            f = open("./data/ProcessedData/EmbeddingVocabIndeces","wb")
            pickle.dump(Vocab,f)
        else :
            f = open("./data/ProcessedData/EmbeddingVocab","rb")
            Vocab = pickle.load(f)
            f = open("./data/ProcessedData/EmbeddingVocabIndeces","rb")
            indeces = pickle.load(f)

        for i in range(len(docs)):
            vector = []
            countEmpty = 0
            for j in range(size):
                if j < len(docs[i]):
                    if docs[i][j] == "":
                        countEmpty+=1
                        continue
                    vector.append(indeces[docs[i][j]]+1)
                else :
                    vector.append(0)
            for j in range(countEmpty):
                vector.append(0)
            vectors.append(vector)
        f = open("./data/ProcessedData/EmbeddingData","wb")
        pickle.dump((vectors,Vocab),f)
    else :
        f = open("./data/ProcessedData/EmbeddingData","rb")
        vectors,Vocab = pickle.load(f)
    return vectors,len(Vocab)+1

def DOC2VEC(docs,vec_size=64):
    vectors = []
    if not os.path.exists('./data/ProcessedData/Doc2VecData'):
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(docs)]
        if not os.path.exists('./data/ProcessedData/Doc2VecModel'):
            max_epochs = 20
            alpha = 0.025
            print("Initiating DOC2VEC")
            model = Doc2Vec(vector_size=vec_size,
                            alpha=alpha,
                            min_alpha=0.00025,
                            min_count=1,
                            dm =1)
            model.build_vocab(tagged_data)
            for epoch in range(max_epochs):
                print('iteration {0}'.format(epoch))
                model.train(tagged_data,
                            total_examples=model.corpus_count,
                            epochs=model.iter)
                # decrease the learning rate
                model.alpha -= 0.0002
                # fix the learning rate, no decay
                model.min_alpha = model.alpha
            model.save("./data/ProcessedData/Doc2VecModel")

        model= Doc2Vec.load("./data/ProcessedData/Doc2VecModel")
        vectors = model.docvecs
        f = open("./data/ProcessedData/Doc2VecData","wb")
        pickle.dump(vectors,f)
    else:
        f = open("./data/ProcessedData/Doc2VecData","rb")
        vectors = pickle.load(f)
    return vectors
