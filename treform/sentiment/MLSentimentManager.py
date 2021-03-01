import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os
import re
from joblib import dump, load
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import treform as ptm

class MachineLearningSentimentAnalyzer:

    def __init__(self):
        name = 'MLSentimentAnalyzer'
        self.reviews_trin_clean = []
        self.reviews_test_clean = []
        self.target=[]

    def split(self):
        reviews_train = []
        for line in open('../data/movie_data/full_train.txt', 'r'):
            reviews_train.append(line.strip())

        reviews_test = []
        for line in open('../data/movie_data/full_test.txt', 'r'):
            reviews_test.append(line.strip())

        self.target = [1 if i < 12500 else 0 for i in range(25000)]

        self.reviews_train_clean = self.preprocess(reviews_train)
        self.reviews_test_clean = self.preprocess(reviews_test)

    def preprocess(self, corpus, language='ko'):
        pipeline = None

        if language == 'ko':
            mecab_path = 'C:\\mecab\\mecab-ko-dic'
            pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                                    ptm.tokenizer.MeCab(mecab_path),
                                    ptm.helper.POSFilter('NN*'),
                                    ptm.helper.SelectWordOnly(),
                                    ptm.ngram.NGramTokenizer(1, 2),
                                    ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                                    )
        elif language == 'en':
            pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                                    ptm.tokenizer.WordPos(),
                                    ptm.helper.POSFilter('NN*'),
                                    ptm.helper.SelectWordOnly(),
                                    ptm.ngram.NGramTokenizer(1, 2),
                                    ptm.helper.StopwordFilter(file='../stopwords/stopwordsEng.txt')
                                    )
        result = pipeline.processCorpus(corpus)
        print('==  ==')

        documents = []
        for doc in result:
            document = ''
            for sent in doc:
                document += " ".join(sent)
            documents.append(document)

        return documents

    def at_least_one(self, without_sw, train):
        threshold = len(train) * 0.01
        reqlist = {k: v for k, v in without_sw.items() if v >= threshold}
        return reqlist

    def at_least_twice(self, words_pos, words_neg, pos_total_dict, neg_total_dict):
        # final_dict_pos={k:v for k,v in words_pos.i
        final_dict = {}
        # print len(words_pos)
        # print len(words_neg)

        for word in words_pos.keys():
            if not word in neg_total_dict:
                final_dict[word] = 1
            elif words_pos[word] >= 2 * neg_total_dict[word]:
                final_dict[word] = 1

        # print len(final_dict)
        for word in words_neg.keys():
            if not word in pos_total_dict:
                final_dict[word] = 1
            elif words_neg[word] >= 2 * pos_total_dict[word]:
                final_dict[word] = 1
        return final_dict

    def remove_stopwords(self, total_dict, stopwords):
        filtereddict = {k: v for k, v in total_dict.items() if k not in stopwords}
        return filtereddict

    def make_customized_feature_vectors(self, train_pos, train_neg, test_pos, test_neg):
        """
        Returns the feature vectors for all text in the train and test datasets.
        """
        # Determine a list of words that will be used as features.
        # This list should have the following properties:
        #   (1) Contains no stop words
        #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
        #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
        pos_total_dict = {}
        for line in train_pos:
            pos_word_dict = {}
            for word in line:
                pos_word_dict[word] = 1
            for word in pos_word_dict.keys():
                if word in pos_total_dict:
                    pos_total_dict[word] = pos_total_dict[word] + 1
                else:
                    pos_total_dict[word] = 1

        neg_total_dict = {}
        for line in train_neg:
            neg_word_dict = {}
            for word in line:
                neg_word_dict[word] = 1
            for word in neg_word_dict.keys():
                if word in neg_total_dict:
                    neg_total_dict[word] = neg_total_dict[word] + 1
                else:
                    neg_total_dict[word] = 1
        print(len(pos_total_dict))
        print(len(neg_total_dict))

        at_least_one_words_pos = self.at_least_one(pos_total_dict, train_pos)
        at_least_one_words_neg = self.at_least_one(neg_total_dict, train_neg)
        # print len(at_least_one_words_pos)
        # print len(at_least_one_words_neg)
        final_dict = self.at_least_twice(at_least_one_words_pos, at_least_one_words_neg, pos_total_dict, neg_total_dict)

        # Using the above words as features, construct binary vectors for each text in the training and test set.
        # These should be python lists containing 0 and 1 integers.
        feature_list = final_dict.keys()

        train_pos_vec = []
        train_neg_vec = []
        test_pos_vec = []
        test_neg_vec = []

        for line in train_pos:
            line_dict = {}
            for word in line:
                line_dict[word] = 1
            vector_list = []
            for word in feature_list:
                if word in line_dict:
                    vector_list.append(1)
                else:
                    vector_list.append(0)
            train_pos_vec.append(vector_list)

        for line in train_neg:
            line_dict = {}
            for word in line:
                line_dict[word] = 1
            vector_list = []
            for word in feature_list:
                if word in line_dict:
                    vector_list.append(1)
                else:
                    vector_list.append(0)
            train_neg_vec.append(vector_list)

        for line in test_pos:
            line_dict = {}
            for word in line:
                line_dict[word] = 1
            vector_list = []
            for word in feature_list:
                if word in line_dict:
                    vector_list.append(1)
                else:
                    vector_list.append(0)
            test_pos_vec.append(vector_list)

        for line in test_neg:
            line_dict = {}
            for word in line:
                line_dict[word] = 1
            vector_list = []
            for word in feature_list:
                if word in line_dict:
                    vector_list.append(1)
                else:
                    vector_list.append(0)
            test_neg_vec.append(vector_list)

        # Return the four feature vectors
        return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

    def make_baseline_feature_vectors(self, combined_vectors, labels):
        baseline_vectorizer = CountVectorizer()
        features = baseline_vectorizer.fit_transform(combined_vectors).toarray()

        self.saveVectorizer(baseline_vectorizer)

        print('feature size ' + str(len(features)) + " : " + str(len(labels)))

        X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                            test_size=0.33, random_state=0)

        return X_train, X_test, y_train, y_test

    def make_tfidf_feature_vectors(self, combined_vectors, labels):
        tfidf_vectorizer = TfidfVectorizer()
        features = tfidf_vectorizer.fit_transform(combined_vectors).toarray()
        self.saveVectorizer(tfidf_vectorizer)

        print('feature size ' + str(len(features)) + " : " + str(len(labels)))

        X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                            test_size=0.33, random_state=0)

        return X_train, X_test, y_train, y_test

    def make_feature_vectors(self, features, labels):
        '''
        :param features:
        :param labels:
        :return X_train, X_test, y_train, y_test:
        '''
        print('feature size ' + str(len(features)) + " : " + str(len(labels)))
        X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                            test_size=0.33, random_state=0)

        return X_train, X_test, y_train, y_test

    def build_baseline_model(self, train_X, train_Y, algorithm):
        """
        Returns a ML-based Model that are fit to the training data.
        """
        print(train_X[:10])

        model = None
        if algorithm == 'lr':
            logr_model = LogisticRegression(random_state=0)
            model = logr_model.fit(train_X, train_Y)
        elif algorithm == 'nb':
            gnb = MultinomialNB()
            model = gnb.fit(train_X, train_Y)
        elif algorithm == 'rf':
            rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
            model = rf_classifier.fit(train_X, train_Y)
        elif algorithm == 'svm':
            svm_classifier = LinearSVC()
            model = svm_classifier.fit(train_X, train_Y)

        return model

    def build_models_with_customizedvectors(self, train_pos_X, train_neg_X, algorithm):
        train_Y = ["pos"] * train_pos_X.shape[1] + ["neg"] * train_neg_X.shape[1]
        train_X = train_pos_X + train_neg_X
        print(train_X[:10])

        model = None
        if algorithm == 'lr':
            logr_model = LogisticRegression(random_state=0)
            model = logr_model.fit(train_X, train_Y)
        elif algorithm == 'nb':
            gnb = MultinomialNB()
            model = gnb.fit(train_X, train_Y)
        elif algorithm == 'rf':
            rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
            model = rf_classifier.fit(train_X, train_Y)
        elif algorithm == 'svm':
            svm_classifier = LinearSVC()
            model = svm_classifier.fit(train_X, train_Y)

        return model

    def baselineLR(self):
        baseline_vectorizer = CountVectorizer(binary=True)
        baseline_vectorizer.fit(self.reviews_train_clean)
        X_baseline = baseline_vectorizer.transform(self.reviews_train_clean)
        X_test_baseline = baseline_vectorizer.transform(self.reviews_test_clean)

        X_train, X_val, y_train, y_val = train_test_split(
            X_baseline, self.target, train_size=0.75)

        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            lr = LogisticRegression(C=c)
            lr.fit(X_train, y_train)
            print("Accuracy for C=%s: %s"
                  % (c, accuracy_score(y_val, lr.predict(X_val))))

        final_model = LogisticRegression(C=0.05)
        final_model.fit(X_baseline, self.target)
        print("Final Accuracy: %s"
              % accuracy_score(self.target, final_model.predict(X_test_baseline)))
        # Final Accuracy: 0.88128

    def tfidfLR(self):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(self.reviews_train_clean)
        X = tfidf_vectorizer.transform(self.reviews_train_clean)
        X_test = tfidf_vectorizer.transform(self.reviews_test_clean)

        X_train, X_val, y_train, y_val = train_test_split(
            X, self.target, train_size=0.75)

        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            lr = LogisticRegression(C=c)
            lr.fit(X_train, y_train)
            print("Accuracy for C=%s: %s"
                  % (c, accuracy_score(y_val, lr.predict(X_val))))

        final_tfidf = LogisticRegression(C=1)
        final_tfidf.fit(X, self.target)
        print("Final Accuracy: %s"
              % accuracy_score(self.target, final_tfidf.predict(X_test)))

        # Final Accuracy: 0.882

    def baselineSVM(self):

        baseline_vectorizer = CountVectorizer(binary=True)
        baseline_vectorizer.fit(self.reviews_train_clean)
        X = baseline_vectorizer.transform(self.reviews_train_clean)
        X_test = baseline_vectorizer.transform(self.reviews_test_clean)

        X_train, X_val, y_train, y_val = train_test_split(
            X, self.target, train_size=0.75)

        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            svm = LinearSVC(C=c)
            svm.fit(X_train, y_train)
            print("Accuracy for C=%s: %s"
                  % (c, accuracy_score(y_val, svm.predict(X_val))))

        final_svm_ngram = LinearSVC(C=0.01)
        final_svm_ngram.fit(X, self.target)
        print("Final Accuracy: %s"
              % accuracy_score(self.target, final_svm_ngram.predict(X_test)))

        # Final Accuracy: 0.8974
        self.print(baseline_vectorizer,final_svm_ngram)

    def print(self, vectorizer=None, final=None):
        feature_to_coef = {
            word: coef for word, coef in zip(
                vectorizer.get_feature_names(), final.coef_[0])
        }

        print("Best Positive")
        for best_positive in sorted(feature_to_coef.items(),
                                    key = lambda x: x[1],
                                    reverse = True)[: 30]:
            print(best_positive)

        print("\n\n")
        print("Best Negative")
        for best_negative in sorted(feature_to_coef.items(),
                                    key = lambda x: x[1])[: 30]:
            print(best_negative)

    def evaluate_model(self, model, test_pos_vec, test_neg_vec, print_confusion=False):
        """
        Prints the confusion matrix and accuracy of the model.
        """
        # Use the predict function and calculate the true/false positives and true/false negative.
        # YOUR CODE HERE
        predict1=model.predict(test_pos_vec).tolist()
        predict2=model.predict(test_neg_vec).tolist()
        #print predict1[0:5]
        #print type(predict1) is list
        #sys.exit(0)
        tp=predict1.count('pos')
        fn=predict1.count('neg')
        tn=predict2.count('neg')
        fp=predict2.count('pos')
        accuracy=float((tp+tn))/(len(test_pos_vec)+len(test_neg_vec))

        if print_confusion:
            print("predicted:\tpos\tneg")
            print("actual:")
            print("pos\t\t%d\t%d" % (tp, fn))
            print("neg\t\t%d\t%d" % (fp, tn))
        print("accuracy: %f" % (accuracy))

    def save(self, model, model_name='sentiment.model'):
        dump(model, model_name)

    def saveVectorizer(self, vectorizer, model_name='senti_vectorizer.model'):
        dump(vectorizer, model_name)

    def load(self, model_name):
        return load(model_name)

    def loadVectorizer(self, model_name='senti_vectorizer.model'):
        return load(model_name)

    def predict_sample(self, model, vectorizer_model):
        docs = ["한국 경제 글로벌 위기 수요 위축 시장 경제 붕귀 자동차 수출 빨간불 내수 촉진 증진 방향성 제고",
                "밝기 5등급 정도 도심 밖 맨눈 충분히 관측 가능 새해 미국인 8월 행운 기대",
                "최순실 민간인 국정농단 의혹 사건 진상규명 국정조사 특별위원회 1차 청문회 이재용 삼성전자 부회장 재벌 총수 9명 증인 출석"]

        text_features = vectorizer_model.transform(docs)
        predictions = model.predict(text_features)
        for text, predicted in zip(docs, predictions):
            print('"{}"'.format(text))
            print("  - Predicted as: '{}'".format(str(predicted)))
            print("")

    def predict(self, docs, model, vectorizer_model):
        text_features = vectorizer_model.transform(docs)
        predictions = model.predict(text_features)
        for text, predicted in zip(docs, predictions):
            print('"{}"'.format(text))
            print("  - Predicted as: '{}'".format(str(predicted)))
            print("")

        return predictions