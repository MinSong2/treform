import treform as ptm
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    corpus = ptm.CorpusFromFieldDelimitedFile('../sample_data/donald.txt',2)
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1,2),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                            )
    result = pipeline.processCorpus(corpus)

    print(result)

    import re
    documents = []
    for doc in result:
        for sent in doc:

            sentence = ' '.join(sent)
            sentence = re.sub('[^A-Za-z0-9가-힣_ ]+', '', sentence)
            sentence = sentence.strip()
            print(sentence)
            if len(sentence) > 0:
                documents.append(sentence)

    print(len(documents))
    co = ptm.cooccurrence.CooccurrenceWorker()
    co_result, vocab = co(documents)

    graph_builder = ptm.graphml.GraphMLCreator()
    #mode is either with_threshold or without_threshod
    mode='with_threshold'
    if mode is 'without_threshold':
        print(str(co_result))
        print(str(vocab))
        graph_builder.createGraphML(co_result, vocab, "test1.graphml")
    elif mode is 'with_threshold':
        cv = CountVectorizer()
        cv_fit = cv.fit_transform(documents)
        word_list = cv.get_feature_names();
        count_list = cv_fit.toarray().sum(axis=0)
        word_hist = dict(zip(word_list, count_list))

        print(str(co_result))
        print(str(word_hist))

        graph_builder.createGraphMLWithThreshold(co_result, word_hist, vocab, "test.graphml",threshold=35.0)
        display_limit=50
        graph_builder.summarize_centrality(limit=display_limit)
        title = '동시출현 기반 그래프'
        file_name='test.png'
        graph_builder.plot_graph(title,file=file_name)
