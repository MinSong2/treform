
from krwordrank.word import KRWordRank
from joblib import load

class BaseKeywordExtraction:
    IN_TYPE = [str]
    OUT_TYPE = [str]

class TextRankExtractor(BaseKeywordExtraction):
    def __init__(self, pos_tagger_name=None, mecab_path=None,
                 lang='ko', max=10,
                 stopwords=[], combined_keywords=False):
        import treform.keyword.textrank as tr
        self.inst = tr.TextRank(pos_tagger_name=pos_tagger_name,mecab_path=mecab_path,lang=lang,stopwords=stopwords)
        self.max=max
        self.combined_keywords = combined_keywords
    def __call__(self, *args, **kwargs):
        import nltk.tokenize
        sents = nltk.tokenize.sent_tokenize(*args)
        for sent in sents:
            self.inst.build_keywords(sent)
        return self.inst.get_keywords(self.max,self.combined_keywords)


class MLKeyphraseExtractor(BaseKeywordExtraction):
    def __init__(self, language='ko', model_name='../../models/svm_keyphrase.model'):
        self.model = load(model_name)
        self.language=language

    def __call__(self, *args, **kwargs):

        from treform.keyword.ml_keyword_builder import get_features
        from treform.keyword.ml_keyword_extractor import extract_candidate_keywords, \
            extract_candidate_keywords_for_training

        self.OUT_TYPE = [list, str]
        feature_list = []

        candidates = extract_candidate_keywords_for_training(args[0], language=self.language)
        set_candidates = set(candidates)
        unique_list = list(set_candidates)
        if len(candidates) > 0:
            feature_list.extend([(get_features(args[0], key, candidates, 0, language=self.language)) for key in set_candidates])
        #for features in feature_list:
        #print(set_candidates)
        #print(feature_list)
        preds = self.model.classify_many(feature_list)
        #self.model.prob_classify_many(feature_list)
        labels = self.model.labels()
        #print(labels)
        print(preds)
        keyphrases = []
        for i, each_pred in enumerate(preds):
            if each_pred == 1:
                keyphrase = unique_list[i]
                keyphrases.append(keyphrase)
        return keyphrases

class TextRankSummarizer(BaseKeywordExtraction):
    def __init__(self,pos_tagger_name=None,mecab_path=None,max=3):
        import treform.keyword.textrank as tr
        self.inst=tr.TextRank(pos_tagger_name=pos_tagger_name,mecab_path=mecab_path)
        self.max=max

    def __call__(self, *args, **kwargs):
        return self.inst.summarize(args[0],self.max)

class KeywordExtractionKorean(BaseKeywordExtraction):
    def __init__(self, min_count=2, max_length=10,
                 beta=0.85, max_iter=10, verbose=True, num_words=20):
        self.min_count=min_count
        self.max_length=max_length
        self.beta=beta
        self.max_iter=max_iter
        self.verbose=verbose
        self.num_words=num_words

        self.inst=KRWordRank(min_count, max_length,self.verbose)

    def __call__(self, *args, **kwargs):
        _num_keywords=10
        #print(str(args[0]) + "\n")
        keywords, rank, graph = self.inst.extract(args[0], self.beta, self.max_iter, self.num_words)

        return keywords