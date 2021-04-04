
from treform.keyword.ml_keyword_builder import train

language = 'ko' #either ko or en
doc_path = '../sample_data/keyphrase_training_data/korean/KeaRaw'
key_path = '../sample_data/keyphrase_training_data/korean/KeaTrain'
model_path = '../models/ko_svm_keyphrase.model'

language = 'en'
doc_path = '../sample_data/keyphrase_training_data/english/documents/'
key_path = '../sample_data/keyphrase_training_data/english/teams/'
model_path = '../models/en_svm_keyphrase.model'

train(language=language, doc_path=doc_path, key_path=key_path, model_path=model_path)


