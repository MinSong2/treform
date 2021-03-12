import treform as ptm
import platform
from nltk.probability import FreqDist
import re, operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pylab as pl
from wordcloud import WordCloud

def plotFdist(n, title=None, fdist=None, fontprop=None):
    fsort_tuple = sorted(fdist.items(), key=operator.itemgetter(1),
                         reverse=True)
    freqs_np_vals = np.array([t[1] for t in fsort_tuple])[0:n]
    freqs_np_words = np.array([t[0] for t in fsort_tuple])[0:n]

    pl.rcParams['figure.figsize'] = (20.0, 6.0)
    width = .35
    ind = np.arange(len(freqs_np_vals))
    plt.title(title, fontproperties=fontprop)
    plt.bar(ind, freqs_np_vals)
    plt.xticks(ind + width / 2, freqs_np_words, rotation='vertical', fontproperties=fontprop)
    plt.show()

def wordOnlyFDist(fdist):
    # only leave letters (does the rest count as "language"? tricky)
    word_only_keys = [k for k in fdist.keys() if re.search(r'^[가-힣a-zA-Z]+$',
                                                           k)]
    return ({key: fdist[key] for key in word_only_keys})


print(str(platform.system()).lower())
if str(platform.system()).lower().startswith('win'):
    # Window의 경우 폰트 경로
    font_path = 'C:/Windows/Fonts/malgun.ttf'
elif str(platform.system()).lower().startswith('mac'):
    #for Mac
    font_path='/Library/Fonts/AppleGothic.ttf'

pipeline = ptm.Pipeline(ptm.splitter.KoSentSplitter(),
                        ptm.tokenizer.Komoran(),
                        #ptm.tokenizer.WordPos(),
                        ptm.helper.POSFilter('NN*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt'),
                        ptm.counter.WordCounter())

corpus = ptm.CorpusFromFile('../data/sampleKor.txt')

result = pipeline.processCorpus(corpus)

print(result)
print()

doc_collection = ''
term_counts = {}
for doc in result:
    for sent in doc:
        for _str in sent:
            term_counts[_str[0]] = term_counts.get(_str[0], 0) + int(_str[1])
            freq = range(int(_str[1]))
            co = ''
            for n in freq:
                co +=  ' ' + _str[0]
            doc_collection += ' ' + co

term_fdist = FreqDist()
word_freq = []
for key, value in term_counts.items():
    word_freq.append((value,key))
    term_fdist[key] += value

fontprop = fm.FontProperties(fname=font_path)
NUM_PLOT = 200
plotFdist(n=NUM_PLOT, title='Corpus', fdist=wordOnlyFDist(term_fdist), fontprop=fontprop)

word_freq.sort(reverse=True)
print(word_freq)

f = open("demo_result.txt", "w", encoding='utf8')
for pair in word_freq:
    f.write(pair[1] + '\t' + str(pair[0]) + '\n')
f.close()

# Generate a word cloud image
wordcloud = WordCloud().generate(doc_collection)

# lower max_font_size
wordcloud = WordCloud(font_path=font_path,
                      max_font_size=40,
                      background_color='white',
                      collocations=False)

wordcloud.generate(doc_collection)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
