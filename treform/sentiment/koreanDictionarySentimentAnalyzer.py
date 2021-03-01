# -*- coding: UTF-8 -*-

import io
import re


class DictionarySentimentAnalyzer:
    def __init__(self):
        name = 'DictionarySentimentAnalyzer'
        self.dict_list={}

    def readPolarityDictionary(self, file):
        fh = open(file, mode="r", encoding="utf-8")
        fh.readline()
        for line in fh:
            fields = line.split(',')
            n_token = ''
            toks = fields[0].split(";")
            for tok in toks:
                n_token += tok.split("/")[0]
            print("element " + n_token + " -- " + fields[len(fields)-2] + " : " + fields[len(fields)-1])
            sent_dict = {}
            sent_dict['word'] = n_token
            sent_dict['polarity'] = fields[len(fields)-2]
            if sent_dict['polarity'] is 'NEG':
                sent_dict['score'] = -float(fields[len(fields)-1].replace("\n",""))
            elif sent_dict['polarity'] is 'POS':
                sent_dict['score'] = float(fields[len(fields) - 1].replace("\n", ""))
            else:
                sent_dict['score'] = 0.0

            if n_token not in self.dict_list:
                self.dict_list[n_token] = sent_dict

    def readKunsanUDictionary(self, file):
        with open(file, encoding='utf-8-sig', mode='r') as f:
            for line in f:
                fields = line.split('\t')
                escaped = fields[0]
                sent_dict = {}
                if len(fields) < 2: continue

                if escaped not in self.dict_list:
                    sent_dict['word'] = escaped
                    sent_dict['score'] = float(fields[1])
                    print('term ' + escaped + " : " + fields[1])
                    if float(fields[1]) < 0:
                        sent_dict['polarity'] = 'NEG'
                    elif float(fields[1]) > 0:
                        sent_dict['polarity'] = 'POS'
                    else:
                        sent_dict['polarity'] = 'NEU'

                    self.dict_list[escaped] = sent_dict

    def readCurseDictionary(self, file):
        with open(file, encoding='utf-8-sig', mode='r') as f:
            for line in f:
                term = line.strip()
                sent_dict={}
                if term not in self.dict_list:
                    sent_dict['word'] = term
                    sent_dict['score'] = -2.0
                    print('term ' + term)
                    sent_dict['polarity'] = 'NEG'

                    self.dict_list[term] = sent_dict

    def readNegativeDictionary(self, file):
        with open(file, encoding='utf-8-sig', mode='r') as f:
            for line in f:
                term = line.strip()
                sent_dict={}
                if term not in self.dict_list:
                    sent_dict['word'] = term
                    sent_dict['score'] = -1.0
                    print('term ' + term)
                    sent_dict['polarity'] = 'NEG'

                    self.dict_list[term] = sent_dict

    def readPositiveDictionary(self, file):
        with open(file, encoding='utf-8-sig', mode='r') as f:
            for line in f:
                term = line.strip()
                sent_dict={}
                if term not in self.dict_list:
                    sent_dict['word'] = term
                    sent_dict['score'] = 1.0
                    print('term ' + term)
                    sent_dict['polarity'] = 'POS'
                    self.dict_list[term] = sent_dict

    def readPositiveEmotiDictionary(self, file):
        with open(file, encoding='utf-8-sig', mode='r') as f:
            for line in f:
                term = line.strip()
                sent_dict={}
                if term not in self.dict_list:
                    sent_dict['word'] = term
                    sent_dict['score'] = 1.0
                    print('term ' + term)
                    sent_dict['polarity'] = 'POS'
                    self.dict_list[term] = sent_dict

    def readNegativeEmotiDictionary(self, file):
        with open(file, encoding='utf-8-sig', mode='r') as f:
            for line in f:
                term = line.strip()
                sent_dict={}
                if term not in self.dict_list:
                    sent_dict['word'] = term
                    sent_dict['score'] = -1.0
                    print('term ' + term)
                    sent_dict['polarity'] = 'NEG'
                    self.dict_list[term] = sent_dict

    def getSentiDictionary(self):
        return self.dict_list

if __name__ == '__main__':
    import pyTextMiner as ptm
    import io
    import nltk

    sentiAnalyzer = DictionarySentimentAnalyzer()

    file_name = './data/SentiWord_Dict.txt'
    sentiAnalyzer.readKunsanUDictionary(file_name)
    file_name = './data/korean_curse_words.txt'
    sentiAnalyzer.readCurseDictionary(file_name)
    file_name = './data/negative_words_ko.txt'
    sentiAnalyzer.readNegativeDictionary(file_name)
    file_name = './data/positive_words_ko.txt'
    sentiAnalyzer.readPositiveDictionary(file_name)
    file_name = './data/emo_negative.txt'
    sentiAnalyzer.readNegativeEmotiDictionary(file_name)
    file_name = './data/emo_positive.txt'
    sentiAnalyzer.readPositiveEmotiDictionary(file_name)
    file_name = './data/polarity.csv'
    sentiAnalyzer.readPolarityDictionary(file_name)

    dict_list = sentiAnalyzer.getSentiDictionary()

    pipeline = None
    corpus = ptm.CorpusFromFieldDelimitedFile('../data/donald.txt',2)
    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    mode = 'korean_lemmatizer'
    if mode is not 'korean_lemmatizer':
         pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                             ptm.tokenizer.MeCab(mecab_path),
                             #ptm.tokenizer.Komoran(),
                             ptm.helper.SelectWordOnly(),
                             #ptm.ngram.NGramTokenizer(1,2,concat=' '),
                             ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt'))
    else :
         pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                             ptm.tokenizer.MeCab(mecab_path),
                             #ptm.tokenizer.Komoran(),
                             ptm.lemmatizer.SejongPOSLemmatizer(),
                             ptm.helper.SelectWordOnly(),
                             #ptm.ngram.NGramTokenizer(1, 2, concat=' '),
                             ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt'))

    documents = ['오늘은 비가와서 그런지 매우 우울하다',
                 'KBO 일명 잘나가는 야구선수들 대부분 팬서비스는 뒷전... 니들 연봉은 팬들이 주는거다!!',
                 '시험이 끝나야 놀지 스트레스 받아ㅠㅠ',
                 '꽤 잘 만들어진 좀비 영화라는 생각이 듬',
                 '옷을 잔뜩 사서 완전 무거워서 신나',
                 '행복한 하루의 끝이라 않좋네!',
                 '더운날에는 아이스커피가 무섭지 않다~~']

    #result = pipeline.processCorpus(corpus)
    result = pipeline.processCorpus(documents)
    print(len(corpus.docs))

    for doc in result:
        total_score = 0.0
        count = 0

        for sent in doc:
            idx = 0
            for _str in sent:
                if len(_str) > 0:
                    score = 0.0
                    flag = False
                    if len(_str.split(' ')) > 0:
                        if _str.split(' ')[0] == '안':
                            _str = _str.split(' ')[1]
                            flag = True
                    dictionary_ele = dict_list.get(_str)
                    if (dictionary_ele != None):
                        polarity = dictionary_ele.get('polarity')
                        if flag:
                            score = -float(dictionary_ele.get('score'))
                        else:
                            score = float(dictionary_ele.get('score'))

                        count += 1
                        #print(_str + " == " + polarity + " " + str(score))
                        total_score += score
                    else:
                        total_score += score

                idx+=1

        if (count != 0):
            avg_score = total_score/count
            print("AVG SCORE " + str(avg_score) + " for " + str(doc))