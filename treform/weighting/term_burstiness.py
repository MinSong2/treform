import time
import pandas as pd

# stats
import numpy as np

# Visualisation
import matplotlib.pyplot as plt

# monitoring
import time

# data cleaning
import re

# lemmatisation
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# stopwords
from nltk.corpus import stopwords

# Clustering
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy

# Machine learning
import sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import roc_curve, auc

ngram_length = 3
min_yearly_df = 5

long_ma_length = 12
short_ma_length = 6
signal_line_ma = 3
significance_ma_length = 3

significance_threshold = 0.0002
years_above_significance = 3
testing_period = 3

# Detection threshold is set such that the top 500 terms are chosen
burstiness_threshold_prediction = 0.003
burstiness_threshold_detection = 0.002451

plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rc('font', family='sans-serif')

year_range = list(range(1988,2018))

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
htmltags = '<[^>]+>'
htmlspecial = '&#?[xX]?[a-zA-Z0-9]{2,8};'

start_delimiter = 'documentstart'
sent_delimiter = 'sentenceboundary'
end_delimiter = 'documentend'

delimiters = [start_delimiter, sent_delimiter, end_delimiter]

# Download the lemmatisesr
wnl = WordNetLemmatizer()

# Create a tokeniser
count = CountVectorizer(strip_accents='ascii', min_df=1)
tokeniser = count.build_analyzer()

def normalise_acronymns(text):
    '''
    Remove the periods in acronyms.
    Adapted from the method found at https://stackoverflow.com/a/40197005
    '''
    return re.sub(r'(?<!\w)([A-Z, a-z])\.', r'\1', text)

def normalise_decimals(text):
    '''
    Remove the periods in decimal numbers and replace with POINT
    '''
    return re.sub(r'([0-9])\.([0-9])', r'\1POINT\2', text)


def split_into_sentences(text):
    '''
    Sentence splitter adapted from https://stackoverflow.com/a/31505798
    '''
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)

    # my addition
    text = re.sub(htmltags, " ", text)
    text = re.sub(htmlspecial, " ", text)

    if "Ph.D" in text:
        text = text.replace("Ph.D.", "PhD")

    text = re.sub("\s" + alphabets + "[.] ", " \\1", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1\\2\\3", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1\\2", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1 \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1", text)
    text = re.sub(" " + alphabets + "[.]", " \\1", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")

    text = text.replace(".", "<stop>")
    text = text.replace("?", "<stop>")
    text = text.replace("!", "<stop>")

    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]

    non_empty = []
    for s in sentences:
        # we require that there be two alphanumeric characters in a row
        if len(re.findall("[A-Za-z0-9][A-Za-z0-9]", s)) > 0:
            non_empty.append(s)
    return non_empty


def pad_sentences(sentences):
    '''
    Takes a list of sentences and returns a string in which:
        - The beginning of the abstract is indicated by DOCUMENTSTART
        - The end is indicated by DOCUMENTEND
        - Sentence boundaries are indicated by SENTENCEBOUNDARY

    The number of delimiters used is dependent on the ngram length
    '''
    sent_string = (' ' + (sent_delimiter + ' ') * (ngram_length - 1)).join(sentences)

    return (start_delimiter + ' ') * (ngram_length - 1) + sent_string + (' ' + end_delimiter) * (ngram_length - 1)


def cleaning_pipeline(title, abstract):
    '''
    Takes a binary string and returns a list of cleaned sentences, stripped of punctuation and lemmatised
    '''

    title = normalise_decimals(normalise_acronymns(title.decode()))
    abstract = normalise_decimals(normalise_acronymns(abstract.decode()))
    sentences = [title] + split_into_sentences(abstract)

    # strip out punctuation and make lowercase
    clean_sentences = []
    for s in sentences:
        # Deal with special cases
        s = re.sub(r'[-/]', ' ', s)

        # Remove all other punctuation
        s = re.sub(r'[^\w\s]', '', s)

        clean_sentences.append(s.lower())

    # pad sentences with delimiters
    text = pad_sentences(clean_sentences)

    # Lemmatise word by word
    lemmas = []
    for word in tokeniser(text):
        lemmas.append(wnl.lemmatize(word))

    return ' '.join(lemmas)


def calc_macd(dataset):
    long_ma = dataset.ewm(span=long_ma_length).mean()
    short_ma = dataset.ewm(span=short_ma_length).mean()
    significance_ma = dataset.ewm(span=significance_ma_length).mean()
    macd = short_ma - long_ma
    signal = macd.ewm(span=signal_line_ma).mean()
    hist = macd - signal
    return long_ma, short_ma, significance_ma, macd, signal, hist


def calc_significance(stacked_vectors, significance_threshold, n):
    # Must have been above the significance threshold for two consecutive timesteps
    a = stacked_vectors > significance_threshold
    b = a.rolling(window=n).sum()
    return stacked_vectors[stacked_vectors.axes[1][np.where(b.max() >= n)[0]]]


def calc_burstiness(hist, scaling_factor):
    return hist.iloc[long_ma_length - 1:] / scaling_factor


def calc_scaling(significance_ma, method):
    if method == "max":
        scaling = significance_ma.iloc[significance_ma_length - 1:].max()
    elif method == "mean":
        scaling = significance_ma.iloc[significance_ma_length - 1:].mean()
    elif method == "sqrt":
        scaling = np.sqrt(significance_ma.iloc[significance_ma_length - 1:].max())
    return scaling

def max_burstiness(burstiness, absolute=False):
    if absolute:
        b = pd.concat([np.abs(burstiness).max(), burstiness.idxmax()], axis=1)
    else:
        b = pd.concat([burstiness.max(), burstiness.idxmax()], axis=1)
    b.columns = ["max", "location"]
    return b


def feature_selection(dataset):
    '''
    Compile the features for the prediction step
    '''
    long_ma = dataset.ewm(span=long_ma_length).mean()
    short_ma = dataset.ewm(span=short_ma_length).mean()
    significance_ma = dataset.ewm(span=significance_ma_length).mean()
    macd = short_ma - long_ma
    signal = macd.ewm(span=signal_line_ma).mean()
    hist = macd - signal

    scaling_factor = calc_scaling(significance_ma, "sqrt")
    burstiness_over_time = calc_burstiness(hist, scaling_factor)
    burstiness = max_burstiness(burstiness_over_time)

    X = long_ma.iloc[long_ma_length:].T
    scaled_hist = hist.iloc[long_ma_length:] / scaling_factor
    scaled_signal = signal.iloc[long_ma_length:] / scaling_factor

    Xtra = pd.concat([significance_ma.iloc[-1],
                      dataset.iloc[-1],
                      significance_ma.iloc[significance_ma_length:].std() / scaling_factor,
                      significance_ma.iloc[significance_ma_length:].max(),
                      significance_ma.iloc[significance_ma_length:].min(),
                      scaling_factor
                      ], axis=1)
    X = pd.concat([X, scaled_hist.T, scaled_signal.T, Xtra], axis=1)

    X.columns = [str(i) for i in range(8)] + ["hist" + str(i) for i in range(8)] + ["signal" + str(i) for i in
                                                                                    range(8)] + [
                    "significance",
                    "prevalence",
                    "scaled std",
                    "max",
                    "min",
                    "scaling"
                ]

    return X

def balanced_subsample(x,y,subsample_size=1.0):
    # from https://stackoverflow.com/a/23479973
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys


vocab = set()
for year in range(1988, 2018):
    df = pd.read_csv('../Data/clean_dblp/' + str(year) + '.csv')

    # The same as above, applied year by year instead.
    t0 = time.time()

    vectorizer = CountVectorizer(strip_accents='ascii',
                                 ngram_range=(1, ngram_length),
                                 min_df=min_yearly_df)

    vector = vectorizer.fit_transform(df.text)

    # Save the new words
    vocab = vocab.union(vectorizer.vocabulary_.keys())

    print(year, len(vocab), time.time() - t0)

vocabulary = {}
i = 0
for v in vocab:
    # Remove delimiters
    if start_delimiter in v:
        pass
    elif end_delimiter in v:
        pass
    elif sent_delimiter in v:
        pass
    else:
        vocabulary[v] = i
        i += 1

print(len(vocabulary.keys()))

vectors = []
for year in range(1988, 2018):
    df = pd.read_csv('../Data/clean_dblp/' + str(year) + '.csv')

    # The same as above, applied year by year instead.
    t0 = time.time()

    vectorizer = CountVectorizer(strip_accents='ascii',
                                 ngram_range=(1, ngram_length),
                                 vocabulary=vocabulary)

    vectors.append(vectorizer.fit_transform(df.text))

    print(year, time.time() - t0)

summed_vectors = []

for y in range(len(vectors)):
    vector = vectors[y]

    # Set all elements that are greater than one to one -- we do not care if a word is used multiple times in
    # the same document
    vector[vector > 1] = 1

    # Sum the vector along columns
    summed = np.squeeze(np.asarray(np.sum(vector, axis=0)))

    # Normalise by dividing by the number of documents in that year
    normalised = summed / vector.shape[0]

    # Save the summed vector
    summed_vectors.append(normalised)

# Stack vectors vertically, so that we have the full history of popularity/time for each term
stacked_vectors = np.stack(summed_vectors, axis=1)

print(stacked_vectors.shape)

stacked_vectors = pd.DataFrame(stacked_vectors.transpose(), columns=list(vocabulary.keys()))

import pickle

stacked_vectors = pickle.load(open('../Data/methods_paper/stacked_vectors.p', "rb"))
burstvectors = pickle.load(open('../Data/methods_paper_2/burstvectors_500.p', "rb"))

normalisation = stacked_vectors.sum(axis=1)
stacked_vectors = stacked_vectors.divide(normalisation, axis='index')*100

stacked_vectors = calc_significance(stacked_vectors, significance_threshold, years_above_significance)
print(stacked_vectors.shape)

#Calculate burstiness
long_ma, short_ma, significance_ma, macd, signal, hist = calc_macd(stacked_vectors)
scaling_factor = calc_scaling(significance_ma, "sqrt")
burstiness_over_time = calc_burstiness(hist, scaling_factor)
burstiness = max_burstiness(burstiness_over_time)

#Set a threshold such that the top 500 bursty terms are included
print(np.sum(burstiness["max"]>0.002451))

bursts = list(burstiness["max"].index[np.where(burstiness["max"]>burstiness_threshold_detection)[0]])
print(bursts)

#Cluster bursts based on co-occurence
#We cluster our 500 bursts based on their co-occurence in abstracts

# vectorise again, using these terms only
vectors = []
for year in range(1988, 2018):
    df = pd.read_csv('../Data/clean_dblp/' + str(year) + '.csv')

    # The same as above, applied year by year instead.
    t0 = time.time()

    vectorizer = CountVectorizer(strip_accents='ascii',
                                 ngram_range=(1, ngram_length),
                                 vocabulary=bursts)

    vector = vectorizer.fit_transform(df.text)

    # If any element is larger than one, set it to one
    vector.data = np.where(vector.data > 0, 1, 0)

    vectors.append(vector)

    print(year, time.time() - t0)

cooccurrence = []
for v in vectors:
    c = v.T * v
    c.setdiag(0)
    c = c.todense()
    cooccurrence.append(c)

all_cooccurrence = np.sum(cooccurrence, axis=0)

# Translate co-occurence into a distance
dists = 1 - all_cooccurrence / all_cooccurrence.max()

# Remove the diagonal (squareform requires diagonals be zero)
dists -= np.diag(np.diagonal(dists))

# Put the distance matrix into the format required by hierachy.linkage
flat_dists = squareform(dists)

# Get the linkage matrix
linkage_matrix = hierarchy.linkage(flat_dists, "ward")

assignments = hierarchy.fcluster(linkage_matrix, 80, 'maxclust')
print(len(bursts))
print(len(set(assignments)))

clusters = defaultdict(list)

for term, assign in zip(bursts, assignments):
    clusters[assign].append(term)

for key in sorted(clusters.keys()):
    print(key, ':', ', '.join(clusters[key]))

#Graph selected bursty terms over time
#We manually remove clusters that contain copyright declarations, etc. Then we filter down to 52, choosing a representative sample over time. We choose one or two terms to represent each cluster.
df = pd.read_csv('clusters3/clusters.csv')
clusters = [d.split(', ') for d in df.terms]


def get_prevalence(cluster):
    indices = []
    for term in cluster:
        indices.append(bursts.index(term))

    prevalence = []
    for year in range(30):
        prevalence.append(
            100 * np.sum(np.sum(burstvectors[year][:, indices], axis=1) > 0) / burstvectors[year].shape[0])

    return prevalence


yplots = 13
xplots = 4
fig, axs = plt.subplots(yplots, xplots)
plt.subplots_adjust(right=1, hspace=0.9, wspace=0.3)
plt.suptitle('Prevalence of selected bursty clusters over time', fontsize=14)
fig.subplots_adjust(top=0.95)
fig.set_figheight(16)
fig.set_figwidth(12)
x = np.arange(0, 30)

prevalences = []
for i, cluster in enumerate(clusters):
    prevalence = get_prevalence(cluster)
    prevalences.append(prevalence)
    title = df.name[i]
    axs[int(np.floor((i / xplots) % yplots)), i % xplots].plot(x, prevalence, color='k', ls='-', label=title)
    axs[int(np.floor((i / xplots) % yplots)), i % xplots].grid()
    ymax = np.ceil(max(prevalence) * 2) / 2
    if ymax == 0.5 and max(prevalence) < 0.25:
        ymax = 0.25
    elif ymax == 2.5:
        ymax = 3
    axs[int(np.floor((i / xplots) % yplots)), i % xplots].set_ylim(0, ymax)
    axs[int(np.floor((i / xplots) % yplots)), i % xplots].set_xlim(0, 30)
    axs[int(np.floor((i / xplots) % yplots)), i % xplots].set_title(title, fontsize=12)

    if i % yplots != yplots - 1:
        axs[i % yplots, int(np.floor((i / yplots) % xplots))].set_xticklabels([])
    else:
        axs[i % yplots, int(np.floor((i / yplots) % xplots))].set_xticklabels([1988, 1998, 2008, 2018])

axs[6, 0].set_ylabel('Percentage of documents containing term (%)', fontsize=12)




