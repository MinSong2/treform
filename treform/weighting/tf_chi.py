import math
import numpy as np

def tf_chi(corpus,package):
	labelset = package ["labelset"]
	weights = package ["weights"]
	doclist = package ["doclist"]
	doclist = dict(doclist)
	dictlist = {}
	doclen = {}
	totaldoc = len(corpus)
	worddict = {}

	for i in corpus:
		labell = i["label"]
		docl = i["document"]
		doclen[docl] = i["length"]
		for j in i["split_sentence"].split():

			if labell not in dictlist:
				dictlist[labell] = {}
			if docl not in dictlist[labell]:
				dictlist[labell][docl] = {}
			if j not in dictlist[labell][docl]:
				dictlist[labell][docl][j] = 1
			else:
				dictlist[labell][docl][j] += 1

			# doclist : label —— word ——　doc set
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

	# chi-square
	for labell in labelset:
		for word in doclist[labell]:
			a_b = worddict[word]
			a = len(doclist[labell][word])
			b = a_b - a
			c_d = sum([(worddict[x]) for x in worddict.keys() if x!=word])
			c = sum([len(doclist[labell][x]) for x in (doclist[labell]) if x!=word])
			d = c_d - c

			if (a_b*c_d*(b+d)*(a+c)) != 0:
				weights[labell][word] = totaldoc*1.0* (a*d-b*c)* (a*d-b*c) /(a_b*c_d*(b+d)*(a+c))

	#print chi
	# tf-chi
	tf_chi = {}
	for labell in labelset:
		tf_chi[labell] = {}
		for doc in dictlist[labell]:
			tf_chi[labell][doc] = {}
			for word in dictlist[labell][doc]:
				tf_chi[labell][doc][word] = dictlist[labell][doc][word]*1.0 / doclen[doc]
				if word in weights[labell]:
					tf_chi[labell][doc][word] *= weights[labell][word]

	package ["labelset"] = labelset
	package ["weights"] = weights
	package ["doclist"] = doclist

	return tf_chi
