from treform.document_classification.ml_textclassification import documentClassifier
import treform as ptm

if __name__ == '__main__':
    document_classifier = documentClassifier()
    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab(mecab_path),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(2, 2),
                            #ptm.tokenizer.LTokenizerKorean(),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                            )

    ml_algorithms = ['RandomForestClassifier', 'LinearSVC', 'MultinomialNB', 'LogisticRegression', 'KNN',
                     'SGDClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier']
    # model_name = 0  -- RandomForestClassifier
    # model_name = 1  -- LinearSVC
    # model_name = 2  -- MultinomialNB
    # model_name = 3  -- LogisticRegression
    # model_name = 4  -- KNN
    # model_name = 5  -- SGDClassifier
    # model_name = 6 -- DecisionTreeClassifier
    # model_name = 7 -- AdaBoostClassifier
    model_index = 7
    model_name = ml_algorithms[7]

    #document category and id map
    id_category_json = '../models/ml_id_category.json'

    #mode is either train or predict
    mode = 'train'
    if mode is 'train':
        input_file ='../sample_data/3_class_naver_news.csv'
        # 1. text processing and representation
        corpus = ptm.CorpusFromFieldDelimitedFileForClassification(input_file,
                                                                   delimiter=',',
                                                                   doc_index=4,
                                                                   class_index=1,
                                                                   title_index=3)
        docs = corpus.docs
        tups = corpus.pair_map
        class_list = []
        for id in tups:
            #print(tups[id])
            class_list.append(tups[id])

        result = pipeline.processCorpus(corpus)
        print('==  ==')

        documents = []
        for doc in result:
            document = ''
            for sent in doc:
                document += " ".join(sent)
            documents.append(document)

        document_classifier.preprocess(documents,class_list,id_category_json=id_category_json)

        X_train, X_test, y_train, y_test, y_pred, indices_test, model = document_classifier.train(model_index=6)

        print('training is finished')

        document_classifier.evaluate(y_test,y_pred,indices_test,model)
        document_classifier.save(model, model_name='../model/' + model_name + '.model')
        document_classifier.saveVectorizer(model_name='../model/' + model_name +  '_vectorizer.model')

    elif mode is 'predict':
        model=document_classifier.load('../model/' + model_name + '.model')
        vectorizer_model=document_classifier.loadVectorizer(model_name='../model/' + model_name +  '_vectorizer.model')
        document_classifier.predict(model,vectorizer_model)

        #7. prediction
        input = "../sample_data/navernews.txt"
        corpus = ptm.CorpusFromFieldDelimitedFile(input,3)

        result = pipeline.processCorpus(corpus)
        print('==  ==')

        documents = []
        for doc in result:
            document = ''
            for sent in doc:
                document += " ".join(sent)
            documents.append(document)

        document_classifier.predict_realtime(model,vectorizer_model, documents, id_category_json=id_category_json)
