import math
import numpy as np

def tf_dc(corpus,package):
	labelset = package ["labelset"]
	weights = package ["weights"]
	voca = package ["voca"]

	label_text = {}
	doclen = {}
	dictlist = {}

	for i in corpus:
		labell = i["label"]
		sentence = i["split_sentence"].split()
		if labell not in label_text:
			label_text[labell] = []
		label_text[labell] = label_text[labell] + sentence

	for a in label_text:
		listt = label_text[a]
		doclen[a]= len(listt)
		if a not in dictlist:
			dictlist[a] = {}
		for i in listt:
			if i not in dictlist[a]:
				dictlist[a][i] = 0
			dictlist[a][i] += 1

	for word in voca:
		hlist = np.zeros((len(labelset)))
		#print len(hlist)
		for cate in range(len(labelset)):
			if word in dictlist[labelset[cate]]:
				hlist[cate] = dictlist[labelset[cate]][word]
		#print hlist
		hlist = hlist/np.sum(hlist)
		for i in range(len(hlist)):
			if abs(hlist[i]-0.0)<1e-5:
				hlist[i]=1
		weights[word] = 1.0 + ( np.sum( hlist*np.log2(hlist)) ) / (math.log(len(labelset),2))

	for i in dictlist:
		for j in dictlist[i]:
			dictlist[i][j] = dictlist[i][j] * 1.0 / doclen[i]
			if j in weights and math.isnan(weights[j]) is not True \
					and math.isnan(dictlist[i][j]) is not True:
				dictlist[i][j] = dictlist[i][j] * weights[j]
	package ["labelset"] = labelset
	package ["weights"] = weights
	package ["voca"] = voca
	return dictlist