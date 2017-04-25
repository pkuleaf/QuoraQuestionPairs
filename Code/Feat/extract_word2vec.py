#!/usr/bin/env python
# encoding: utf-8

import cPickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
from gensim.models import KeyedVectors


def tokenize(s):
    s = str(s)
    try:
        us=s.decode('utf-8')
    except:
        us=s
    words = word_tokenize(us)
    return words

def get_words(values, words):
    for i, q in tqdm(enumerate(values)):
        tokens = tokenize(q)
        for t in tokens:
            words[t]=True
    return words

if __name__ == '__main__':
    fword2vec = '../../Data/GoogleNews-vectors-negative300.bin'
    data_dir = '../../Output/'
    fdata = ['train.processed.csv', 'test.processed.csv']
    foutput = 'word2vec.pkl'

    print 'Loading word2vec model'
    model=KeyedVectors.load_word2vec_format(fword2vec, binary=True)

    print 'Getting tokenized words'
    words = {}
    for fn in fdata:
        fn = data_dir + fn
        data = pd.read_csv(fn)
        get_words(data.question1.values, words)
        get_words(data.question2.values, words)

    print 'Getting vectors for presented words'
    for word in words.keys():
        if word in model:
            words[word] = model[word]
        else:
            del words[word]

    print 'Save output'
    cPickle.dump(words, open(data_dir+foutput, 'wb'), protocol=2)
