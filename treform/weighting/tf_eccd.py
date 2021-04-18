import math
import numpy as np

def tf_eccd(corpus,package):

	labelset = package ["labelset"]
	weights = package ["weights"]
	doclist = package ["doclist"]
	doclist = dict(doclist)
	dictlist = {}
	doclen = {}
	n = len(corpus)
	worddict = {}

	for i in corpus:
		labell = i["label"]
		docl = i["document"]
		if labell not in doclen:
			doclen[labell] = 0
		doclen[labell] += i["length"]
		for j in i["split_sentence"].split():
			if labell not in dictlist:
				dictlist[labell] = {}
			if j not in dictlist[labell]:
				dictlist[labell][j] = 1
			else:
				dictlist[labell][j] += 1

			if labell not in doclist:
				doclist[labell] = {}
			if j not in doclist[labell]:
				doclist[labell][j] = set()
			doclist[labell][j].add(docl)

	entropy = {}
	tf = {}
	for labell in labelset:
		tf[labell] = {}
		for word in doclist[labell]:
			if word not in worddict:
				worddict[word] = 0
			worddict[word] += len(doclist[labell][word])
			total = sum([dictlist[x][word]  for x in dictlist if word in dictlist[x]])
			tf[labell][word] = dictlist[labell][word]*1.0/total

	Emax = -999
	for labell in labelset:
		for word in doclist[labell]:
			if word not in entropy:
				entropy[word] = -sum([ tf[x][word]*1.0*math.log(tf[x][word],2) for x in labelset if word in tf[x] ])
				Emax = max(entropy[word],Emax)

	for labell in labelset:
		weights[labell] = {}
		for word in doclist[labell]:
			a_b = worddict[word]
			a = len(doclist[labell][word])
			b = a_b - a
			c_d = sum([(worddict[x]) for x in worddict.keys() if x!=word])
			c = sum([len(doclist[labell][x]) for x in (doclist[labell]) if x!=word])
			d = c_d - c
			if ((a*d) - (b*c)) != 0 and Emax != 0:
				weights[labell][word] = ((a*d-b*c)*1.0/((a+c)*(b+d)))*(Emax - entropy[word])*1.0/Emax

	tf_eccd = {}
	for labell in labelset:
		tf_eccd[labell] = {}
		for word in dictlist[labell]:
			tf_eccd[labell][word] = dictlist[labell][word]*1.0 /  (doclen[labell]*1.0)
			if word in weights[labell]:
				tf_eccd[labell][word] *= weights[labell][word]

	package ["labelset"] = labelset
	package ["weights"] = weights
	package ["doclist"] = doclist

	return tf_eccd
