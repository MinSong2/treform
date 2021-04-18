import math

def tf_idf(corpus, package):
	dictlist = {}
	doclen = {}
	docname = package ["docname"]
	weights = package ["weights"]

	for i in corpus:
		docl = i["document"]
		doclen[docl] = i["length"]
		for j in i["split_sentence"].split():
			# dctlist : doc —— word —— frequency
			if docl not in dictlist:
				dictlist[docl] = {}
			if j not in dictlist[docl]:
				dictlist[docl][j] = 0
			dictlist[docl][j] += 1

			# doclist :  word ——　doc set
			if j not in weights:
				weights[j] = set()
			weights[j].add(docl)
			docname.add(docl)
	for word in weights:
		weights[word] = math.log( ( 1+len(docname)*1.0)/(len(weights[word])*1.0),2)
	tf_idf_weight = {}
	for doc in dictlist:
		tf_idf_weight[doc] = {}
		for word in dictlist[doc]:
			# tf:
			tf_idf_weight[doc][word] = dictlist[doc][word]*1.0 / (doclen[doc]*1.0)
			# tf*idf
			tf_idf_weight[doc][word] *= weights[word]
	package ["docname"] = docname
	package ["weights"] = weights
	#print(tf_idf_weight)

	return tf_idf_weight