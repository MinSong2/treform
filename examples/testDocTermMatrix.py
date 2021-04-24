import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import treform as ptm
import nltk
import string
from collections import defaultdict

def vectorizeCaseZero():
    documents = [
                 'This is the first document.',
                 'This document is the second document.',
                 'And this is the third one.',
                 'Is this the first document?',
                ]

    stem = nltk.stem.SnowballStemmer('english')

    def tokenize(text):
        stem = nltk.stem.SnowballStemmer('english')
        text = text.lower()

        for token in nltk.word_tokenize(text):
            if token in string.punctuation: continue
            yield stem.stem(token)

    def vectorize(doc):
        features = defaultdict(int)
        for token in tokenize(doc):
            features[token] += 1
        return features

    vectors = map(vectorize, documents)

    for doc in vectors:
        for word in doc:
            print(str(word) + ' ' + str(doc[word]))
        print('------------------')

def vectorizeCaseOne():
    documents = [
                 'This is the first document.',
                 'This document is the second document.',
                 'And this is the third one.',
                 'Is this the first document?',
                ]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    print(vectorizer.get_feature_names())
    print(X.toarray())

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    print(vectorizer.get_feature_names())
    print(X.toarray())

def vectorizeCaseTwo():
    corpus = ptm.CorpusFromFieldDelimitedFile('../sample_data/donald.txt',2)

    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(2, 2),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                            )
    result = pipeline.processCorpus(corpus)
    print(result)
    print()

    print('==  ==')

    documents = []
    for doc in result:
        document = ''
        for sent in doc:
            document += " ".join(sent)
        documents.append(document)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    print(vectorizer.get_feature_names())
    print(X.shape)

    print(X.toarray())

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    print(vectorizer.get_feature_names())
    print(len(vectorizer.get_feature_names()))
    print(X.toarray())

vectorizeCaseZero()

#vectorizeCaseOne()

#vectorizeCaseTwo()

