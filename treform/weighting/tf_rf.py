import math
import numpy as np


def tf_rf(corpus,package):
	labelset = package ["labelset"]
	weights = package ["weights"]
	doclist = package ["doclist"]
	doclist = dict(doclist)

	doclen = {}
	dictlist = {}
	worddict = {}
	totaldoc = len(corpus)

	for i in corpus:
		labell = i["label"]
		docl = i["document"]
		doclen[docl] = i["length"]
		if labell not in doclen:
			doclen[labell] = {}
		if docl not in doclen[labell]:
			doclen[labell][docl] = 0
		doclen[labell][docl] += i["length"]

		for j in i["split_sentence"].split():
			# dictlist : label —— doc —— word —— frequency
			if labell not in dictlist:
				dictlist[labell] = {}
			if docl not in dictlist[labell]:
				dictlist[labell][docl] = {}
			if j not in dictlist[labell][docl]:
				dictlist[labell][docl][j] = 0
			dictlist[labell][docl][j] += 1

			# oclist : label —— word ——　doc set
			if labell not in doclist:
				doclist[labell] = {}
			if j not in doclist[labell]:
				doclist[labell][j] = set()
			doclist[labell][j].add(docl)

	for labell in labelset:
		weights[labell] = {}
		for word in doclist[labell]:
			if word not in worddict:
				worddict[word] = 0
			worddict[word] += len(doclist[labell][word])
			a = len(doclist[labell][word])
			c = sum([len(doclist[labell][x]) for x in (doclist[labell]) if x!=word])
			weights[labell][word] = math.log(2+a*1.0/max(1,c*1.0),2)

	tf_rf = {}
	for labell in labelset:
		tf_rf[labell] = {}
		for doc in dictlist[labell]:
			tf_rf[labell][doc] = {}
			for word in dictlist[labell][doc]:
				#print doc + word
				tf_rf[labell][doc][word] = dictlist[labell][doc][word]*1.0 / (doclen[labell][doc]*1.0)
				if word in weights[labell]:
					tf_rf[labell][doc][word] *= weights[labell][word]

	package ["labelset"] = labelset
	package ["weights"] = weights
	package ["doclist"] = doclist
	return tf_rf
