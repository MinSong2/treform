
import treform as ptm
from treform.sentiment.MLSentimentManager import MachineLearningSentimentAnalyzer

sentiAnalyzer = MachineLearningSentimentAnalyzer()

model = sentiAnalyzer.load('sentiment.model')
vectorizer_model = sentiAnalyzer.loadVectorizer(model_name='senti_vectorizer.model')

docs = ['오늘은 세상이 참 아름답게 보이네요! 감사합니다',
        '우울한 날이면 언제나 부정적인 생각에 아프다...']
predictions = sentiAnalyzer.predict(docs, model, vectorizer_model)
for i, predicted in enumerate(predictions):
    print(predicted + ' '
                      '.0.0for ' + docs[i])