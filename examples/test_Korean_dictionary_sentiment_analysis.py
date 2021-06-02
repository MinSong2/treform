import treform as ptm
from treform.sentiment.koreanDictionarySentimentAnalyzer import DictionarySentimentAnalyzer

if __name__ == '__main__':



    sentiAnalyzer = DictionarySentimentAnalyzer()

    file_name = '../data/SentiWord_Dict.txt'
    sentiAnalyzer.readKunsanUDictionary(file_name)
    file_name = '../data/korean_curse_words.txt'
    sentiAnalyzer.readCurseDictionary(file_name)
    file_name = '../data/negative_words_ko.txt'
    sentiAnalyzer.readNegativeDictionary(file_name)
    file_name = '../data/positive_words_ko.txt'
    sentiAnalyzer.readPositiveDictionary(file_name)
    file_name = '../data/emo_negative.txt'
    sentiAnalyzer.readNegativeEmotiDictionary(file_name)
    file_name = '../data/emo_positive.txt'
    sentiAnalyzer.readPositiveEmotiDictionary(file_name)
    file_name = '../data/polarity.csv'
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