import benepar
import nltk
from nltk import tree

class BaseSyntacticParser:
    IN_TYPE = [list, tuple]
    OUT_TYPE = [list, str]

class BeneparSyntacticParser(BaseSyntacticParser):
    def __init__(self):
        try:
            self.parser = benepar.Parser("benepar_ko2")
        except LookupError:
            benepar.download('benepar_ko2')

        if self.parser is None:
            self.parser = benepar.Parser("benepar_ko2")

    def __call__(self, *args, **kwargs):
        words = []
        tags = []
        for tok in args[0]:
            words.append(tok[0])
            tags.append(tok[1])

        input_sentence = benepar.InputSentence(
            words=words,
            tags=tags,
        )
        tree = self.parser.parse(input_sentence)
        #print(tree)
        return tree._pformat_flat("", "()", False)

class NLTKRegexSyntacticParser(BaseSyntacticParser):
    def __init__(self):
        # Define a chunk grammar, or chunking rules, then chunk
        grammar = """
        NP: {<N.*>*<Suffix>?}   # Noun phrase
        VP: {<V.*>*}            # Verb phrase
        AP: {<A.*>*}            # Adjective phrase
        """
        self.parser = nltk.RegexpParser(grammar)

    def __call__(self, *args, **kwargs):
        chunks = self.parser.parse(*args)
        return chunks._pformat_flat("", "()", False)