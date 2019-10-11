import nltk
import re
from nltk.corpus import stopwords
import numpy as np

def Preprocess(docs):
    stops = set(stopwords.words("english"))
    for i in range(len(docs)):
        docs[i] = [w for w in docs[i].lower().split() if not w in stops]
        docs[i] = " ".join(docs[i])
        docs[i] = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", docs[i] )
        docs[i] = re.sub(r"[0-9]", " ", docs[i] )
        docs[i] = re.sub(r"\'s", " 's ", docs[i] )
        docs[i] = re.sub(r"\'ve", " 've ", docs[i] )
        docs[i] = re.sub(r"n\'t", " 't ", docs[i] )
        docs[i] = re.sub(r"\'re", " 're ", docs[i] )
        docs[i] = re.sub(r"\'d", " 'd ", docs[i] )
        docs[i] = re.sub(r"\'ll", " 'll ", docs[i] )
        docs[i] = re.sub(r",", " ", docs[i] )
        docs[i] = re.sub(r"\.", " ", docs[i] )
        docs[i] = re.sub(r"!", " ", docs[i] )
        docs[i] = re.sub(r"\n", " ", docs[i] )
        docs[i] = re.sub(r"\(", "", docs[i] )
        docs[i] = re.sub(r"\)", "", docs[i] )
        docs[i] = re.sub(r"\t", "", docs[i] )
        docs[i] = re.sub(r"\?", " ", docs[i] )
        docs[i] = re.sub(r"\s{2,}", " ", docs[i] )
    return docs
