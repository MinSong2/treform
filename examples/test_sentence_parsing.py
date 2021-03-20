
import nltk
import treform as ptm
from nltk.draw.tree import draw_trees
from nltk import tree, treetransforms
from copy import deepcopy

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Komoran(),
                        ptm.syntactic_parser.BeneparSyntacticParser()
                        )
corpus = ptm.CorpusByDataFrame('../sample_data/parser_sample.txt', '\t', 0, header=False)
#corpus = ptm.CorpusFromFieldDelimitedFile('../sample_data/parser_sample.txt', 0)
print(corpus.docs)

trees = pipeline.processCorpus(corpus)

for tree in trees:
    print(tree[0])
    t = nltk.Tree.fromstring(tree[0])
    draw_trees(t)
