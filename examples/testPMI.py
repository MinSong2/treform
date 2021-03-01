import treform as ptm

corpus=ptm.CorpusFromFile('../data/134963_norm.txt')
pmi=ptm.pmi.PMICalculator(corpus)
sent='노래'
result=pmi.__call__(sent)
print(result)