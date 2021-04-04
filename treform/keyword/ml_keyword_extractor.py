"""
Takes document as input and performs chunking accoring to a pattern to return a list of Candidate Keywords.

Input :
    document - text file
Output :
    list of possible keyphrases

"""
import nltk
from nltk import word_tokenize
from nltk.chunk import RegexpParser
from treform.tokenizer import Komoran

def extract_candidate_keywords(words, tags):
    #Get the words in the document
    #words = word_tokenize(document)
    # Chunk first to get 'Candidate Keywords'
    #tags = nltk.pos_tag(words)

    chunkGram = r""" PHRASE: 
                        {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}
                """

    chunkParser = RegexpParser(chunkGram)
    chunked = chunkParser.parse(tags)

    candidate_keywords = []
    for tree in chunked.subtrees():
        if tree.label() == 'PHRASE':
            candidate_keyword = ' '.join([x for x,y in tree.leaves()])
            candidate_keywords.append(candidate_keyword)

    candidate_keywords = [w for w in candidate_keywords if len(w) > 3 and  len(w.split(' ')) < 6]
    return candidate_keywords


def extract_candidate_keywords_for_training(document, language='ko'):
    if language == "ko":
        komoran = Komoran()
        tags = komoran(document)

        chunkGram = r""" 
                        PHRASE: {<N.*>*<Suffix>?}
                    """
    elif language == "en":
        # Get the words in the document
        words = word_tokenize(document)
        #Chunk first to get 'Candidate Keywords'
        tags = nltk.pos_tag(words)
        chunkGram = r"""
                        PHRASE: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}
                        """
    chunkParser = RegexpParser(chunkGram)
    chunked = chunkParser.parse(tags)

    candidate_keywords = []
    for tree in chunked.subtrees():
        if tree.label() == 'PHRASE':
            candidate_keyword = ' '.join([x for x, y in tree.leaves()])
            candidate_keywords.append(candidate_keyword)

    candidate_keywords = [w for w in candidate_keywords if len(w) > 3 and len(w.split(' ')) < 6]

    return candidate_keywords