
class BaseSynonymExtraction:
    IN_TYPE = [str]
    OUT_TYPE = [str]

class WordNetSynonymExtraction(BaseSynonymExtraction):
    def __init__(self):
        import nltk
        try:
            wnl = nltk.WordNetLemmatizer()
        except LookupError:
            nltk.download('wordnet')

    def __call__(self, *args, **kwargs):
        from nltk.corpus import wordnet
        # Creating a list
        synonyms = []
        for syn in wordnet.synsets(args):
            for lm in syn.lemmas():
                synonyms.append(lm.name())  # adding into synonyms
        print(set(synonyms))
        return synonyms