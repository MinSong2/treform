import treform as ptm
import io

# 다음은 분석에 사용할 corpus를 불러오는 일입니다. sampleEng.txt 파일을 준비해두었으니, 이를 읽어와봅시다.
# ptm의 CorpusFromFile이라는 클래스를 통해 문헌집합을 가져올 수 있습니다. 이 경우 파일 내의 한 라인이 문헌 하나가 됩니다.
#corpus = ptm.CorpusFromFieldDelimitedFile('../sample_data/donald.txt',2)
#corpus = ptm.CorpusFromDirectory('./tmp', True)
#corpus, pair_map = ptm.CorpusFromFieldDelimitedFileWithYear('./data/donald.txt')
corpus = ptm.CorpusFromFile('../sample_data/sampleEng.txt')

#import nltk
#nltk.download('punkt')

#pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
#                        ptm.tokenizer.Komoran(),
#                        ptm.helper.POSFilter('NN*'),
#                        ptm.helper.SelectWordOnly(),
#                        ptm.ngram.NGramTokenizer(3),
#                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
#                        )

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Word())

#pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
#                        ptm.segmentation.SegmentationKorean('../model/korean_segmentation_model.crfsuite'),
#                        ptm.ngram.NGramTokenizer(3),
#                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
#                        )

result = pipeline.processCorpus(corpus)

with io.open("../demofile.csv",'w',encoding='utf8') as f:
    for doc in result:
        for sent in doc:
            f.write('\t'.join(sent) + "\n")

print('== 문장 분리 + 형태소 분석 + 명사만 추출 + 단어만 보여주기 + 구 추출 ==')
print(result)
print()
