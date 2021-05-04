import random

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import treform as ptm
from treform.sentiment.MLSentimentManager import MachineLearningSentimentAnalyzer

labels = ['pos','neg']

language = 'ko'
pipeline = None

_train_negative_docs = []
_test_negative_docs = []
_train_positive_docs = []
_test_positive_docs = []


def read_english_corpus():
    '''
    for English sentiment datasets
    '''

    #_train_negative_docs = ptm.CorpusFromDirectory('txt_sentoken/neg', True)
    #_train_negative_docs = _train_negative_docs.docs[:200]
    #_train_positive_docs = ptm.CorpusFromDirectory('txt_sentoken/pos', True)
    #_train_positive_docs = _train_positive_docs.docs[:200]
    #_test_negative_docs = ptm.CorpusFromDirectory('txt_sentoken/neg', False)
    #_test_negative_docs = _test_negative_docs.docs[:100]
    #_test_positive_docs = ptm.CorpusFromDirectory('txt_sentoken/pos', False)
    #_test_positive_docs = _test_positive_docs.docs[:100]

    _train_negative_docs = ptm.CorpusFromFile('./datasets/twitter/train-neg.txt')
    _train_negative_docs = _train_negative_docs.docs[:5000]
    _train_positive_docs = ptm.CorpusFromFile('./datasets/twitter/train-pos.txt')
    _train_positive_docs = _train_positive_docs.docs[:5000]
    _test_negative_docs = ptm.CorpusFromFile('./datasets/twitter/test-neg.txt')
    _test_negative_docs = _test_negative_docs.docs[:5000]
    _test_positive_docs = ptm.CorpusFromFile('./datasets/twitter/test-pos.txt')
    _test_positive_docs = _test_positive_docs.docs[:5000]

    return _train_negative_docs, _train_positive_docs, _test_negative_docs, _test_positive_docs

def read_korean_corpus():
    '''
    for Korean sentiment datasets
    '''

    corpus = ptm.CorpusFromFieldDelimitedFileForClassification('./datasets/ratings.txt', doc_index=1, class_index=2)
    combined_docs = corpus.docs[1:15000]

    positive_docs = []
    negative_docs = []
    labels = []

    idx = 1
    for doc in combined_docs:
        label = corpus.pair_map[idx]
        label = label.strip()
        labels.append(label)

        if int(label) == 0:
            negative_docs.append(doc)
        elif int(label) == 1:
            positive_docs.append(doc)
        idx += 1

    print(str(len(negative_docs)) + " : " + str(len(positive_docs)))

    neg_train_size = int((len(negative_docs) * 0.8) / len(negative_docs))
    _train_negative_docs = negative_docs[0:neg_train_size]
    _test_negative_docs = negative_docs[neg_train_size+1:len(negative_docs) - 1]

    pos_train_size = int((len(positive_docs) * 0.8) / len(positive_docs))
    _train_positive_docs = positive_docs[0:pos_train_size]
    _test_positive_docs = positive_docs[pos_train_size+1:len(positive_docs) - 1]

    return _train_negative_docs, _train_positive_docs, _test_negative_docs, _test_positive_docs, labels

if language == 'en':
    _train_negative_docs, _train_positive_docs, _test_negative_docs, _test_positive_docs \
                                                                         = read_english_corpus()
elif language == 'ko':
    _train_negative_docs, _train_positive_docs, _test_negative_docs, _test_positive_docs, labels \
                                                                         = read_korean_corpus()
if language == 'ko':
    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(),
                            #ptm.tokenizer.MeCab(mecab_path),
                            ptm.helper.POSFilter('NN*|V*|IC*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1, 2),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                            )
elif language == 'en':
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.WordPos(),
                            ptm.helper.POSFilter('NN*|A*|V*|J*'),
                            ptm.helper.SelectWordOnly(),
                            #ptm.ngram.NGramTokenizer(1, 2),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsEng.txt')
                            )

def make_documents(result):
    docs = []
    for doc in result:
        new_doc = []
        for sent in doc:
            for _str in sent:
                if len(_str) > 0:
                    new_doc.append(_str)
        docs.append(' '.join(new_doc))

    return docs

_train_negative = pipeline.processCorpus(_train_negative_docs)
train_negative = make_documents(_train_negative)
_train_positive = pipeline.processCorpus(_train_positive_docs)
train_positive = make_documents(_train_positive)

_test_negative = pipeline.processCorpus(_test_negative_docs)
test_negative = make_documents(_test_negative)
_test_positive = pipeline.processCorpus(_test_positive_docs)
test_positive = make_documents(_test_positive)

print('==  ==')

sentiAnalyzer = MachineLearningSentimentAnalyzer()
train_pos_vec = []
train_neg_vec = []
test_pos_vec = []
test_neg_vec = []
train_X = []
train_Y = []
test_X = []
test_Y = []

#training_method = customized, baseline
training_method = 'baseline'
if training_method == 'customized':
    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = \
                                    sentiAnalyzer.make_customized_feature_vectors(train_positive,
                                                                                  train_negative,
                                                                                  test_positive,
                                                                                  test_negative)
    features = train_pos_vec + train_neg_vec + test_pos_vec + test_neg_vec
    labels = ["neg"] * len(train_neg_vec) + ["pos"] * len(train_pos_vec) \
             + ["neg"] * len(test_neg_vec) + ["pos"] * len(test_pos_vec)

    train_X, test_X, train_Y, test_Y = \
                                    sentiAnalyzer.make_feature_vectors(features,
                                                                       labels)
elif training_method == 'baseline':

    combined_docs = train_negative + train_positive + test_negative + test_positive
    # random.shuffle(_combined_docs)
    labels = ["neg"] * len(train_negative) + ["pos"] * len(train_positive) + ["neg"] * len(test_negative) + ["pos"] * len(test_positive)

    #vectorizer_name is either count or tfidf (count -- countvectorizer or tfidf -- tfidfvectorizer)
    vectorizer_name = 'tfidf'
    if vectorizer_name == 'count':
        train_X, test_X, train_Y, test_Y = \
                                    sentiAnalyzer.make_baseline_feature_vectors(combined_docs,
                                                                                labels)
    elif vectorizer_name == 'tfidf':
        train_X, test_X, train_Y, test_Y = \
                                    sentiAnalyzer.make_tfidf_feature_vectors(combined_docs,
                                                                                labels)

customized = False
#algorithm is either lr, nb, svm, rf
algorithm = 'svm'

if customized == True:
    model = sentiAnalyzer.build_models_with_customizedvectors(train_pos_vec, train_neg_vec, algorithm)
    sentiAnalyzer.save(model)

    print("ML-based model")
    print("-----------")
    sentiAnalyzer.evaluate_model(model, test_pos_vec, test_neg_vec, True)
    print("")

else:
    model = sentiAnalyzer.build_baseline_model(train_X, train_Y, algorithm)
    sentiAnalyzer.save(model)

    print("ML-based model")
    print("-----------")
    model.fit(train_X, train_Y)
    y_pred = model.predict(test_X)
    conf_mat = confusion_matrix(test_Y, y_pred)
    print(conf_mat)
    print(metrics.classification_report(test_Y, y_pred,
                                        target_names=["neg", "pos"]))
