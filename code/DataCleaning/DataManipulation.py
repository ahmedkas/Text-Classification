import nltk
import re
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import os.path
import tensorflow_hub as hub
import keras
import pickle
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def Manipulations_Selector(docs,option=None,vec_size=512,size=300,features_words=1000,features_chars=5000,word_min = 1,word_max=3,char_min = 2,char_max = 5):
    vocab_size=size
    if option == None:
        print("Please provide the option [USE,DOC2VEC,Embedding]")
    elif option == "USE":
        docs = USE(docs)
    elif option == "DOC2VEC":
        docs = DOC2VEC(docs,vec_size)

    elif option == "WORD2VEC":
        docs = WORD2VEC(docs,vec_size)

    elif option == "WORD2VEC_pre":
        docs,vocab_size = WORD2VEC_pre(docs)

    elif option == "Glove":
        docs, vocab_size= Glove(docs)


    elif option == "Embedding":
        docs,vocab_size = Embedding(docs,size)
    elif option == "n_grams":
        docs= n_grams(docs,features_words,features_chars,word_min,word_max,char_min,char_max)
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


def Embedding(docs, size = 300):
    vectors = []
    if not os.path.exists('./data/ProcessedData/EmbeddingData'+str(size)):
        Vocab = []
        indeces = {}
        for i in range(len(docs)):
            docs[i] = docs[i].split(" ")
            docs[i] = docs[i][:size]
        if not os.path.exists('./data/ProcessedData/EmbeddingVocab'+str(size)):
            for i in range(len(docs)):
                for word in docs[i]:
                    if word != "":
                        Vocab.append(word)
            Vocab = list(set(Vocab))
            indeces = {}
            for i in range(len(Vocab)):
                indeces[Vocab[i]] = i
            print(str(size))
            f = open("./data/ProcessedData/EmbeddingVocab"+str(size),"wb")
            pickle.dump(Vocab,f)
            f = open("./data/ProcessedData/EmbeddingVocabIndeces"+str(size),"wb")
            pickle.dump(indeces,f)
        else :
            f = open("./data/ProcessedData/EmbeddingVocab"+str(size),"rb")
            Vocab = pickle.load(f)
            f = open("./data/ProcessedData/EmbeddingVocabIndeces"+str(size),"rb")
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
        f = open("./data/ProcessedData/EmbeddingData"+str(size),"wb")
        pickle.dump((vectors,Vocab),f)
    else :
        f = open("./data/ProcessedData/EmbeddingData"+str(size),"rb")
        vectors,Vocab = pickle.load(f)

    return vectors,len(Vocab)+1

def DOC2VEC(docs,vec_size=512):
    vectors = []
    if not os.path.exists('./data/ProcessedData/Doc2VecData'+str(vec_size)):
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(docs)]
        if not os.path.exists('./data/ProcessedData/Doc2VecModel'+str(vec_size)):
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
            model.save("./data/ProcessedData/Doc2VecModel"+str(vec_size))

        model= Doc2Vec.load("./data/ProcessedData/Doc2VecModel"+str(vec_size))
        vectors = model.docvecs
        f = open("./data/ProcessedData/Doc2VecData"+str(vec_size),"wb")
        pickle.dump(vectors,f)
    else:
        f = open("./data/ProcessedData/Doc2VecData"+str(vec_size),"rb")
        vectors = pickle.load(f)
    return vectors

def WORD2VEC(docs,vec_size=100):
    w_size =32
    print("Word2Vec size :"+ str(vec_size))
    vectors = []
    if not os.path.exists('./data/ProcessedData/Word2VecData_w'+str(w_size)+"_v"+str(vec_size)):

        sentences =[]
        for item in docs:
            token = item.replace("\t","").replace("\n", "").split(' ')
            sentences.append(token)

        # Convertinf each words in the comments into vec of len 32
        model = Word2Vec(sentences, size=w_size,min_count=1, workers=5, iter=100)
        word_vectors = model.wv

        #padding
        for i in range(len(sentences)):
            if not word_vectors[sentences[i]].shape[0] > vec_size:
                padd = np.zeros((vec_size - word_vectors[sentences[i]].shape[0], w_size))
                padd_sentence = np.concatenate((word_vectors[sentences[i]], padd), axis=0)

            else:
                padd_sentence = word_vectors[sentences[i]][:vec_size]

            vectors.append(padd_sentence)

        print("save word2vec as vectors")
        with open('./data/ProcessedData/Word2VecData_w'+str(w_size)+"_v"+str(vec_size), 'wb') as f:
            pickle.dump(vectors, f)
    else:
        print("Loading saved word2vec vectors")
        f = open('./data/ProcessedData/Word2VecData_w'+str(w_size)+"_v"+str(vec_size),"rb")
        vectors = pickle.load(f)


    return vectors


def Glove(docs):
    embedding_dim = 50
    filepath = './data/glove.6B.50d.txt'

    vectors = []
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(docs)
    docs = tokenizer.texts_to_sequences(docs)
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    word_index = tokenizer.word_index

    # padding
    maxlen = 100
    docs = pad_sequences(docs, padding='post', maxlen=maxlen)
    vocab_size = len(tokenizer.word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]


    for d in docs:
        sentence = []
        for w in d.tolist():
            sentence.append(embedding_matrix[w].tolist())
        vectors.append(sentence)



    return vectors, vocab_size

def WORD2VEC_pre(docs,vec_size=100):
    vec_size =50
    w_size =300
    print("Loading google pre-trained model")
    vectors = []

    # Load vectors directly from the file
    model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    word_vectors = model.wv
    vocab_size = len(word_vectors.vocab)
    sentences =[]

    # print("Tokinizing")

    for item in docs:
        words=[]
        tokens = item.replace("\t","").replace("\n", "").split(' ')

        for t in tokens:
            if t in word_vectors.vocab:
                words.append(t)

        sentences.append(words)


    # print("Applying the model")

    empty_sent = np.zeros((vec_size, w_size))

    #padding
    for i in range(len(sentences)):
        if len(sentences[i]) == 0:
            # print("empyt")
            padd_sentence = empty_sent

        else:

            if not word_vectors[sentences[i]].shape[0] > vec_size:
                padd = np.zeros((vec_size - word_vectors[sentences[i]].shape[0], w_size))
                padd_sentence = np.concatenate((word_vectors[sentences[i]], padd), axis=0)
                # padd_sentence = np.append(word_vectors[sentences[i]], padd)
            else:
                padd_sentence = word_vectors[sentences[i]][:vec_size]

        vectors.append(padd_sentence)

    # print("Vocab size: "+str(vocab_size))
    return vectors, vocab_size


def n_grams(docs,features_words=1000,features_chars=1000,word_min = 2,word_max=4,char_min = 3,char_max = 5):
    print("TF-IDF REp")
    print(features_chars)
    print(features_words)
    vectors = []
    if not os.path.exists('./data/ProcessedData/tf_word_features'):
        print("words TF-IDF Saving ... ")
            # words
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(word_min, word_max),
            max_features=features_words)
        word_vectorizer.fit(docs)
        tf_word_features = word_vectorizer.transform(docs)


        # Save word vocab features as pickle
        f = open("./data/ProcessedData/tf_word_vocab","wb")
        pickle.dump(word_vectorizer.vocabulary_,f)
        # Save word features as pickle
        f = open("./data/ProcessedData/tf_word_features","wb")
        pickle.dump(tf_word_features,f)
    else:
        print("words TF-IDF Loading ...")
        # load word features as pickle
        f = open("./data/ProcessedData/tf_word_features","rb")
        tf_word_features = pickle.load(f)


    if not os.path.exists('./data/ProcessedData/tf_char_features'):
        print("chars TF-IDF Saving ... ")

        #char
        char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            stop_words='english',
            ngram_range=(char_min, char_max),
            max_features=features_chars)
        char_vectorizer.fit(docs)
        tf_char_features = char_vectorizer.transform(docs)

        # Save char vocab features as pickle
        f = open("./data/ProcessedData/tf_char_vocab","wb")
        pickle.dump(char_vectorizer.vocabulary_,f)
        # Save char features as pickle
        f = open("./data/ProcessedData/tf_char_features","wb")
        pickle.dump(tf_char_features,f)
    else:
        print("chars TF-IDF Loading ... ")
        # load char features as pickle
        f = open("./data/ProcessedData/tf_char_features","rb")
        tf_char_features = pickle.load(f)

    vectors = hstack([tf_word_features, tf_char_features])

    vectors = vectors.todense()
    vectors = np.array(vectors)
    return vectors
