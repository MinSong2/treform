import treform as ptm

min_count = 2   # 단어의 최소 출현 빈도수 (그래프 생성 시)
max_length = 20 # 단어의 최대 길이
beta = 0.95
max_iter = 20
verbose = True
num_words=30
keyword_extractor=ptm.keyword.KeywordExtractionKorean(min_count,max_length,beta,max_iter,verbose,num_words)

sents = ['인공지능 또는 AI는 인간의 학습능력, 추론능력, 지각능력, 그외에 인공적으로 구현한 컴퓨터 프로그램 또는 이를 포함한 컴퓨터 시스템이다. 하나의 인프라 기술이기도 하다.[1][2] 인간을 포함한 동물이 갖고 있는 지능 즉, natural intelligence와는 다른 개념이다. 지능을 갖고 있는 기능을 갖춘 컴퓨터 시스템이며, 인간의 지능을 기계 등에 인공적으로 시연(구현)한 것이다. 일반적으로 범용 컴퓨터에 적용한다고 가정한다. 이 용어는 또한 그와 같은 지능을 만들 수 있는 방법론이나 실현 가능성 등을 연구하는 과학 분야를 지칭하기도 한다.',
         '현대사회의 정보기술이 발전함에 따라 정보의 규모는 보다 방대해지고 있으며 정보흐름의 속도 또한 폭발적으로 빨라지면서 대용량의 정보들을 신속하고 정확하게 처리하고 활용하기 위한 논의가 끊임없이 대두되고 있다. 이러한 정보의 처리와 활용의 대표적인 방법론으로 데이터 마이닝을 꼽을 수 있는데, 데이터 마이닝(Data Mining)이란 의사결정 수단을 위하여 대용량의 데이터베이스(database)로부터 의미 있는 규칙과 패턴을 발견하는 기법을 말한다(Hearst, 1999). 데이터 마이닝이 다루는 데이터베이스는 구조에 따라 구조화 데이터베이스(structured database)와 비구조화 데이터베이스(unstructured database)로 구분할 수 있다. 구조화 데이터 또는 정형화 데이터란 매출데이터, 회계데이터 등과 같이 일반적으로 정형화된 수치데이터(numeric data)를 일컫는 것인 반면 비구조화 데이터 또는 비정형화 데이터란 웹상의 블로그 혹은 소셜미디어의 게시물과 같이 수치 데이터가 아닌 문자, 그림이나 영상, 문서처럼 형태와 구조가 복잡한 데이터를 뜻한다. ']
keyword=keyword_extractor(sents)
for word, r in sorted(keyword.items(), key=lambda x: x[1], reverse=True)[:30]:
    print('%8s:\t%.4f' % (word, r))

corpus = ptm.CorpusFromFieldDelimitedFile('../sample_data/donald.txt', 2)

# import nltk
# nltk.download()
# 단어 단위로 분리했으니 이제 stopwords를 제거하는게 가능합니다. ptm.helper.StopwordFilter를 사용하여 불필요한 단어들을 지워보도록 하겠습니다.
# 그리고 파이프라인 뒤에 ptm.stemmer.Porter()를 추가하여 어근 추출을 해보겠습니다.
# 한번 코드를 고쳐서 ptm.stemmer.Lancaster()도 사용해보세요. Lancaster stemmer가 Porter stemmer와 어떻게 다른지 비교하면 재미있을 겁니다.
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Komoran(),
                        ptm.helper.POSFilter('NN*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt'))
result = pipeline.processCorpus(sents)
print(result)
print()

documents=[]
for doc in result:
    document=''
    for sent in doc:
        document += " ".join(sent)
    documents.append(document)

keyword_extractor1=ptm.keyword.KeywordExtractionKorean(min_count,max_length,beta,max_iter,verbose,num_words)
keyword1=keyword_extractor1(documents)
for word, r in sorted(keyword1.items(), key=lambda x: x[1], reverse=True)[:30]:
    print('%8s:\t%.4f' % (word, r))


print('--------------')
print(documents[0])

model_path='../models/svm_keyphrase.model'
keyword_extractor1=ptm.keyword.MLKeyphraseExtractor(language='ko', model_name=model_path)
keyword1=keyword_extractor1(sents[0])
print(keyword1)
