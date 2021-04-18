
from numpy import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def one_hot(data):
	
	values = array(data)
	
	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
	
	# binary encode
	onehot_encoder = OneHotEncoder(sparse = False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	
	return onehot_encoded