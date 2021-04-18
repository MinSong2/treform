

import math
import numpy as np

def iqf_qf_icf(corpus,package):
	weights = package["weights"]
	doclist = {}
	dictlist = {}
	worddict = {}
	word_label_dict = {}
	labelset = package["labelset"]

	n = len(corpus)

	for i in corpus:
		labell = i["label"]
		docl = i["document"]
		if labell not in doclist:
			doclist[labell] = {}
		if labell not in dictlist:
			dictlist[labell] = {}
		if docl not in dictlist[labell]:
			dictlist[labell][docl] = {}

		for j in i["split_sentence"].split():
			#dictlist : label —— doc —— word —— frequency
			if j not in dictlist[labell][docl]:
				dictlist[labell][docl][j] = 1
			else:
				dictlist[labell][docl][j] += 1
			#doclist : label —— word ——　doc set
			if j not in doclist[labell]:
				doclist[labell][j] = set()
			doclist[labell][j].add(docl)

	iqf_qf_icf = {}

	for labell in labelset:
		iqf_qf_icf[labell] = {}
		for word in doclist[labell]:
			if word not in word_label_dict:
				word_label_dict[word] = set()
			word_label_dict[word].add(labell)
			if word not in worddict:
				worddict[word] = 0
			worddict[word] += len(doclist[labell][word])

	#iqf_qf_icf
	for labell in labelset:
		weights[labell] = {}
		for word in doclist[labell]:
			a_b = worddict[word]
			a = len(doclist[labell][word])
			b = a_b - a
			weights[labell][word] = math.log(n*1.0/(a+b),2)*math.log(a+1,2)*math.log( len(labelset)*1.0/len(word_label_dict[word]) +1,2)
	
	package["weights"] = weights
	package["labelset"] = labelset

	return weights
