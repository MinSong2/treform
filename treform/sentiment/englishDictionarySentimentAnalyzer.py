
import treform as ptm
import io
from nltk.corpus import sentiwordnet as swn
import nltk

class EnglishDictionarySentimentAnalyzer:
    def __init__(self):
        name = 'EnglishDictionarySentimentAnalyzer'

    def createDictionary(self):
        nltk.download('sentiwordnet')


if __name__ == '__main__':

    corpus = ptm.CorpusFromFile('../data/sampleEng.txt')
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            #ptm.tokenizer.Word(), #ptm.tokenizer.WordPos()
                            #ptm.helper.StopwordFilter(file='../stopwords/stopwordsEng.txt'),
                            #ptm.tagger.NLTK(),
                            ptm.tokenizer.WordPos(),
                            ptm.lemmatizer.WordNet())

    result = pipeline.processCorpus(corpus)

    EnglishDictionarySentimentAnalyzer().createDictionary()

    for doc in result:
        grand_score = 0.0
        count = 0
        for sent in doc:
            for _str in sent:
                _str[0]
                _str[1]
                pos = ''
                if (str(_str[1]).startswith("N")):
                    pos = 'n'
                elif (str(_str[1]).startswith("J")):
                    pos = 'a'
                elif (str(_str[1]).startswith("V")):
                    pos = 'v'
                try:
                    if (len(pos) > 0):
                        score = 0.0
                        breakdown = swn.senti_synset(str(_str[0]) + '.'+ pos + '.01')
                        #print(str(breakdown) + " " + str(breakdown.pos_score()) + " " + str(breakdown.neg_score()) + " " + str(breakdown.obj_score()))
                        if (breakdown.pos_score() > breakdown.neg_score()):
                            score = breakdown.pos_score()
                            count += 1
                        elif (breakdown.pos_score() < breakdown.neg_score() and breakdown.neg_score() != 0.0):
                            score = -breakdown.neg_score()
                            count += 1
                        grand_score += score
                except:
                    pos = ''

        avg_score = grand_score/count
        print("Average Sentiment Score " + str(avg_score))
