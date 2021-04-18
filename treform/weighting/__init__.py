import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from treform.weighting.iqf_qf_icf import iqf_qf_icf
from treform.weighting.tf_bdc import tf_bdc
from treform.weighting.tf_chi import tf_chi
from treform.weighting.tf_dc import tf_dc
from treform.weighting.tf_eccd import tf_eccd
from treform.weighting.tf_idf import tf_idf
from treform.weighting.tf_ig import tf_ig
from treform.weighting.tf_rf import tf_rf


class BaseWeighting:
    IN_TYPE = [str]
    OUT_TYPE = [list, str]

    def __init__(self):
        self.package = {}
        self.package["voca"] = []
        self.package["labelset"] = []
        self.package["vocafreq"] = {}
        self.package["weights"] = {}
        self.package["doclist"] = []
        self.package["docname"] = set()

    def preprocess(self, input, label_list=[]):
        vocafreq = self.package["vocafreq"]
        labelset = self.package["labelset"]
        corpus = []
        doccount = {}

        label = 'UNI'
        for i, _doc in enumerate(input):
            document = {}
            if len(label_list) > 0:
                document["label"] = label_list[i]
            else:
                document["label"] = label
            if label not in doccount:
                doccount[label] = 0
            doccount[label] += 1
            docname = label + str(doccount[label])
            document["document"] = docname

            document["split_sentence"] = _doc

            if label not in labelset:
                labelset.append(label)

            document["length"] = len(_doc)
            corpus.append(document)

        cv = CountVectorizer()
        cv_fit = cv.fit_transform(input)
        word_list = cv.get_feature_names()

        count_list = np.asarray(cv_fit.sum(axis=0))[0]
        vocabfreq = dict(zip(word_list, count_list))
        self.package["vocafreq"] = vocabfreq
        self.package["voca"] = word_list
        self.package["labelset"] = labelset

        #print(vocabfreq)

        return corpus

class TfIdf(BaseWeighting):
    def __init__(self, input, label_list=[]):
        super().__init__()
        self.corpus = super().preprocess(input, label_list=label_list)

    def __call__(self):
        weights = tf_idf(self.corpus, self.package)
        print('----------tf-idf--------')
        return weights

class TfBdc(BaseWeighting):
    def __init__(self, input, label_list=[]):
        super().__init__()
        self.corpus = super().preprocess(input, label_list=label_list)

    def __call__(self):
        weights = tf_bdc(self.corpus, self.package)
        print('----------tf-dbc--------')
        return weights

class IqfQfIcf(BaseWeighting):
    def __init__(self, input, label_list=[]):
        super().__init__()
        self.corpus = super().preprocess(input, label_list=label_list)

    def __call__(self):
        weights = iqf_qf_icf(self.corpus, self.package)
        print('----------iqf-qf-icf--------')
        return weights

class TfChi(BaseWeighting):
    def __init__(self, input, label_list=[]):
        super().__init__()
        self.corpus = super().preprocess(input, label_list=label_list)

    def __call__(self):
        weights = tf_chi(self.corpus, self.package)
        print('----------tf-chi--------')
        return weights

class TfDc(BaseWeighting):
    def __init__(self, input, label_list=[]):
        super().__init__()
        self.corpus = super().preprocess(input, label_list=label_list)

    def __call__(self):
        weights = tf_dc(self.corpus, self.package)
        print('----------tf-dc--------')
        return weights

class TfEccd(BaseWeighting):
    def __init__(self, input, label_list=[]):
        super().__init__()
        self.corpus = super().preprocess(input, label_list=label_list)

    def __call__(self):
        weights = tf_eccd(self.corpus, self.package)
        print('----------tf-eccd--------')
        return weights

class TfIg(BaseWeighting):
    def __init__(self, input, label_list=[]):
        super().__init__()
        self.corpus = super().preprocess(input, label_list=label_list)

    def __call__(self):
        weights = tf_ig(self.corpus, self.package)
        print('----------tf-ig--------')
        return weights

class TfRf(BaseWeighting):
    def __init__(self, input, label_list=[]):
        super().__init__()
        self.corpus = super().preprocess(input, label_list=label_list)

    def __call__(self):
        weights = tf_rf(self.corpus, self.package)
        print('----------tf-rf--------')
        return weights