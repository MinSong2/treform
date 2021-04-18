import treform as ptm

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(2, 2),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                            )

corpus = ptm.CorpusFromFieldDelimitedFile('../data/donald.txt',2)

result = pipeline.processCorpus(corpus)

#print(result)
print()

print('==  ==')

documents = []
for doc in result:
    document = ' '
    for sent in doc:
        for word in sent:
            if len(word) > 0:
                document += ' ' + word
    document = document.strip()
    if len(document) > 0:
       documents.append(document)

#print(documents)

weight_algorithm = 'tf_idf'

if weight_algorithm == 'tf_idf':
    tf_idf = ptm.weighting.TfIdf(documents)
    weights = tf_idf()

elif weight_algorithm == 'tf_bdc':
    tf_bdc = ptm.weighting.TfBdc(documents)
    weights = tf_bdc()

elif weight_algorithm == 'iqf_qf_icf':
    iqf_qf_icf = ptm.weighting.IqfQfIcf(documents)
    weights = iqf_qf_icf()

elif weight_algorithm == 'tf_chi':
    tf_chi = ptm.weighting.TfChi(documents)
    weights = tf_chi()

elif weight_algorithm == 'tf_dc':
    tf_dc = ptm.weighting.TfDc(documents)
    weights = tf_dc()

elif weight_algorithm == 'tf_eccd':
    tf_eccd = ptm.weighting.TfEccd(documents)
    weights = tf_eccd()

elif weight_algorithm == 'tf_ig':
    tf_ig = ptm.weighting.TfIg(documents)
    weights = tf_ig()

elif weight_algorithm == 'tf_rf':
    tf_rf = ptm.weighting.TfRf(documents)
    weights = tf_rf()

print(weights)
