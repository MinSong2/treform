import pickle
from os import listdir
from nltk.corpus import stopwords
from pickle import dump
from string import punctuation
import os, sys
from stat import *
import treform as ptm
import io

class sentimentDataManager:
    def __init__(self):
        name = 'sentimentDataManager'

    # load doc into memory
    def load_doc(self, filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

    # save a dataset to file
    def save_dataset(self, dataset, filename):
        dump(dataset, open(filename, 'wb'))
        print('Saved: %s' % filename)

    def load_dataset(self, filename):
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model

if __name__ == '__main__':

    _negative_docs = ptm.CorpusFromDirectory('txt_sentoken/neg', True)
    _positive_docs = ptm.CorpusFromDirectory('txt_sentoken/pos', True)

    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Word(),
                            ptm.helper.StopwordFilter(file='../stopwordsEng.txt'),
                            ptm.stemmer.Porter())
    _neg_result = pipeline.processCorpus(_negative_docs)
    _pos_result = pipeline.processCorpus(_positive_docs)
    print('== Splitting Sentence + Tokenizing + Stopwords Removal + Stemming : Porter ==')
    print(_neg_result)
    print()

    negative_docs = list()
    for doc in _neg_result:
        new_doc = []
        for sent in doc:
            for _str in sent:
                if len(_str) > 0:
                    new_doc.append(_str)
        negative_docs.append(' '.join(new_doc))

    positive_docs = list()
    for doc in _pos_result:
        new_doc = []
        for sent in doc:
            for _str in sent:
                if len(_str) > 0:
                    new_doc.append(_str)
        positive_docs.append(' '.join(new_doc))


    # load all training reviews
    trainX = negative_docs + positive_docs
    print("TRAIN X " + str(len(negative_docs)))

    trainy = [0 for _ in range(900)] + [1 for _ in range(900)]
    sentimentDataManager().save_dataset([trainX, trainy], 'train.pkl')

    # load all test reviews
    _negative_docs = ptm.CorpusFromDirectory('txt_sentoken/neg', False)
    _positive_docs = ptm.CorpusFromDirectory('txt_sentoken/pos', False)

    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Word(),
                            ptm.helper.StopwordFilter(file='../stopwordsEng.txt'),
                            ptm.stemmer.Porter())
    _neg_result = pipeline.processCorpus(_negative_docs)
    _pos_result = pipeline.processCorpus(_positive_docs)
    print('== Splitting Sentence + Tokenizing + Stopwords Removal + Stemming : Porter ==')
    print(_neg_result)
    print()

    negative_docs = list()
    for doc in _neg_result:
        new_doc = []
        for sent in doc:
            for _str in sent:
                if len(_str) > 0:
                    new_doc.append(_str)
        negative_docs.append(' '.join(new_doc))

    positive_docs = list()
    for doc in _pos_result:
        new_doc = []
        for sent in doc:
            for _str in sent:
                if len(_str) > 0:
                    new_doc.append(_str)
        positive_docs.append(' '.join(new_doc))

    testX = negative_docs + positive_docs
    testY = [0 for _ in range(100)] + [1 for _ in range(100)]
    sentimentDataManager().save_dataset([testX, testY], 'test.pkl')
