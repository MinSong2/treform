from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

class sentimentPredictor:
    def __init__(self):
        name = 'sentimentPredictor'

    # load a clean dataset
    def load_dataset(self, filename):
        return load(open(filename, 'rb'))

    # fit a tokenizer
    def create_tokenizer(self, lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer


    # calculate the maximum document length
    def max_length(self, lines):
        return max([len(s.split()) for s in lines])


    # encode a list of lines
    def encode_text(self, tokenizer, lines, length):
        # integer encode
        encoded = tokenizer.texts_to_sequences(lines)
        # pad encoded sequences
        padded = pad_sequences(encoded, maxlen=length, padding='post')
        return padded

if __name__ == '__main__':
    # load datasets
    trainLines, trainLabels = sentimentPredictor().load_dataset('train.pkl')
    testLines, testLabels = sentimentPredictor().load_dataset('test.pkl')

    # create tokenizer
    tokenizer = sentimentPredictor().create_tokenizer(trainLines)
    # calculate max document length
    length = sentimentPredictor().max_length(trainLines)
    # calculate vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Max document length: %d' % length)
    print('Vocabulary size: %d' % vocab_size)
    # encode data
    trainX = sentimentPredictor().encode_text(tokenizer, trainLines, length)
    testX = sentimentPredictor().encode_text(tokenizer, testLines, length)
    print(trainX.shape, testX.shape)

    # load the model
    model = load_model('model.h5')

    # evaluate model on training dataset
    loss, acc = model.evaluate([trainX, trainX, trainX], array(trainLabels), verbose=0)
    print('Train Accuracy: %f' % (acc * 100))

    # evaluate model on test dataset dataset
    loss, acc = model.evaluate([testX, testX, testX], array(testLabels), verbose=0)
    print('Test Accuracy: %f' % (acc * 100))

    """
   tokenizer1 = sentimentPredictor().create_tokenizer(['this is a good news'])
   # calculate max document length
   length1 = sentimentPredictor().max_length(['this is a good news'])
   # calculate vocabulary size
   vocab_size1 = len(tokenizer1.word_index) + 1
   print('Max document length: %d' % length1)
   print('Vocabulary size: %d' % vocab_size1)
   # encode data
   test = sentimentPredictor().encode_text(tokenizer1, ['this is a good news'], length1)
   """

    # make a prediction
    test_predict = model.predict([testX, testX, testX], verbose=1, batch_size=1)

    print(test_predict)
