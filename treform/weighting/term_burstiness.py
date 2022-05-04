import platform
from copy import deepcopy

from datetime import datetime, timedelta
from dateutil import parser

import pandas as pd
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt

import time
from collections import defaultdict

from matplotlib.legend_handler import HandlerLine2D
from sklearn.feature_extraction.text import CountVectorizer

# Machine learning
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

import tomotopy as tm

import pickle
import os

ngram_length = 3
min_yearly_df = 5

# 하이퍼 파라미터 지정.
long_ma_length = 4
short_ma_length = 2
signal_line_ma = 2
significance_ma_length = 2

significance_threshold = 0.000001
years_above_significance = 3
testing_period = 3

# Detection threshold is set such that the top 500 terms are chosen
burstiness_threshold_prediction = 0.003
burstiness_threshold_detection = 0.002451

plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rc('font', family='sans-serif')

year_range = list(range(1988,2020))

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
    a = stacked_vectors>significance_threshold
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
    print('burst over time ' + str(len(burstiness_over_time)))

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

    pd.options.display.max_columns = None
    XY = pd.concat([X, scaled_hist.T, scaled_signal.T, Xtra], axis=1)

    print(scaled_hist.T.info(10))
    print('++++++++++++++++++')
    print(scaled_signal.T.info(10))
    print('++++++++++++++++++')
    print(Xtra.info(10))
    print('++++++++++++++++++')
    print(X.info(10))
    print('=======================')
    print(XY.info(10))

    XY.columns = [str(i) for i in range(2)] + ["hist" + str(i) for i in range(2)] + ["signal" + str(i) for i in
                                                                                    range(2)] + [
                    "significance",
                    "prevalence",
                    "scaled std",
                    "max",
                    "min",
                    "scaling"
                ]

    return XY

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


def get_prevalence(cluster, bursts, burstvectors, unique_time_stamp):
    indices = []
    for term in cluster:
        if term in bursts:
            indices.append(bursts.index(term))

    prevalence = []
    for year in unique_time_stamp:
        prevalence.append(
            100 * np.sum(np.sum(burstvectors[year][:, indices], axis=1) > 0) / burstvectors[year].shape[0])

    return prevalence

def compute_term_burstiness(dataset):
    # Build a vocabulary
    # We have to build a vocabulary before we vectorise the data. This is because we want to set limits on the size of the vocabulary.

    vocab = set()

    df = pd.read_csv(dataset)

    grouped_df = df.groupby(df.columns[0])

    for key, item in grouped_df:
        a_group = grouped_df.get_group(key)

        print(a_group.head(10))
        # The same as above, applied year by year instead.
        t0 = time.time()

        vectorizer = CountVectorizer(min_df=min_yearly_df)
        print(a_group.iloc[:, 4].head(10))

        vector = vectorizer.fit_transform(a_group.iloc[:, 4])

        # Save the new words
        vocab = vocab.union(vectorizer.vocabulary_.keys())
        time_stamp = a_group.iloc[:, 0].tolist()[0]
        print(time_stamp, len(vocab), time.time() - t0)

    vocabulary = {}
    i = 0
    for v in vocab:
        vocabulary[v] = i
        i += 1

    print(len(vocabulary.keys()))

    # Go year by year and vectorise based on our vocabulary
    # We read in the cleaned data and vectorise it according to our vocabulary.
    vectors = []
    grouped_df = df.groupby(df.columns[0])

    for key, item in grouped_df:
        a_group = grouped_df.get_group(key)

        # The same as above, applied year by year instead.
        t0 = time.time()

        vectorizer = CountVectorizer(vocabulary=vocabulary)

        vectors.append(vectorizer.fit_transform(a_group.iloc[:, 4]))
        time_stamp = a_group.iloc[:, 0].tolist()[0]
        print(time_stamp, time.time() - t0)

    # Summing the vectors
    # We sum the vectors along columns, so that we have the popularity of each term in each year.

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

    normalisation = stacked_vectors.sum(axis=1)
    stacked_vectors = stacked_vectors.divide(normalisation, axis='index') * 100

    stacked_vectors = calc_significance(stacked_vectors, significance_threshold, years_above_significance)
    print(stacked_vectors.shape)

    # Calculate burstiness
    long_ma, short_ma, significance_ma, macd, signal, hist = calc_macd(stacked_vectors)
    scaling_factor = calc_scaling(significance_ma, "sqrt")
    burstiness_over_time = calc_burstiness(hist, scaling_factor)
    burstiness = max_burstiness(burstiness_over_time)

    # Set a threshold such that the top 500 bursty terms are included
    print(np.sum(burstiness["max"] > 0.002451))

    bursts = list(burstiness["max"].index[np.where(burstiness["max"] > burstiness_threshold_detection)[0]])
    print(bursts)

    # Cluster bursts based on co-occurence
    # vectorise again, using these terms only
    burstvectors = {}
    grouped_df = df.groupby(df.columns[0])

    corpus = tm.utils.Corpus()

    unique_time_stamp = []
    for key, item in grouped_df:
        a_group = grouped_df.get_group(key)
        # The same as above, applied year by year instead.
        t0 = time.time()

        vectorizer = CountVectorizer(vocabulary=bursts)
        vector = vectorizer.fit_transform(a_group.iloc[:, 4])

        for i, _doc in enumerate(a_group.iloc[:, 4].tolist()):
            corpus.add_doc(_doc.strip().split())
            if i % 10 == 0:
                print('Document #{} has been loaded'.format(i))

        # If any element is larger than one, set it to one
        vector.data = np.where(vector.data > 0, 1, 0)
        time_stamp = a_group.iloc[:, 0].tolist()[0]
        unique_time_stamp.append(time_stamp)
        burstvectors[time_stamp] = vector

        print(time_stamp, time.time() - t0)

    with open('../models/unique_time_stamp.pickle', 'wb') as handle:
        pickle.dump(unique_time_stamp, handle)

    clusters = defaultdict(list)

    model = tm.LDAModel(k=30, min_cf=10, min_df=5, rm_top=50, corpus=corpus)
    model.optim_interval = 20
    model.burn_in = 200
    model.train(0)

    # Let's train the model
    for i in range(0, 1500, 20):
        print('Iteration: {:04} LL per word: {:.4}'.format(i, model.ll_per_word))
        model.train(20)
    print('Iteration: {:04} LL per word: {:.4}'.format(1000, model.ll_per_word))

    # extract candidates for auto topic labeling
    extractor = tm.label.PMIExtractor(min_cf=10, min_df=10, max_len=5, max_cand=10000, normalized=True)
    cands = extractor.extract(model)
    labeler = tm.label.FoRelevance(model, cands, min_df=5, smoothing=1e-2, mu=0.25)

    cluster_label = {}
    for k in range(model.k):
        print('Topic #{}'.format(k))
        m = 0
        for label, score in labeler.get_topic_labels(k, top_n=1):
            if m > 0:
                continue

            print("Labels:", label)
            cluster_label[k] = label
            m += 1

        for word, prob in model.get_topic_words(k, top_n=20):
            print('\t', word, prob, sep='\t')
            clusters[k].append(word)

    for key in sorted(clusters.keys()):
        print(key, ':', ', '.join(clusters[key]))

    font_path = ''
    if platform.system() is 'Windows':
        # Window의 경우 폰트 경로
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    elif platform.system() is 'Darwin':
        # for Mac
        font_path = '/Library/Fonts/AppleGothic.ttf'

    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rc('axes', unicode_minus=False)

    total = model.k
    y_val = int(np.ceil(total / 4))

    # Graph selected bursty terms over time
    # yplots = 13
    # xplots = 4
    yplots = y_val
    xplots = 4
    fig, axs = plt.subplots(yplots, xplots)
    plt.subplots_adjust(right=1, hspace=0.9, wspace=0.3)
    plt.suptitle('Prevalence of selected bursty clusters over time', fontsize=14)
    fig.subplots_adjust(top=0.95)
    fig.set_figheight(16)
    fig.set_figwidth(12)

    print(bursts[0])
    print(burstvectors[0])

    prevalences = []
    for i, cluster in enumerate(clusters):
        prevalence = get_prevalence(clusters[cluster], bursts, burstvectors, unique_time_stamp)

        x = range(0, len(prevalence))

        prevalences.append(prevalence)
        title = cluster_label[cluster]
        axs[int(np.floor((i / xplots) % yplots)), i % xplots].plot(x, prevalence, color='k', ls='-', label=title)
        axs[int(np.floor((i / xplots) % yplots)), i % xplots].grid()
        ymax = np.ceil(max(prevalence) * 2) / 2
        if ymax == 0.5 and max(prevalence) < 0.25:
            ymax = 0.25
        elif ymax == 2.5:
            ymax = 3
        axs[int(np.floor((i / xplots) % yplots)), i % xplots].set_ylim(0, ymax)
        axs[int(np.floor((i / xplots) % yplots)), i % xplots].set_xlim(0, len(prevalence))
        axs[int(np.floor((i / xplots) % yplots)), i % xplots].set_title(title, fontsize=12)

        if i % yplots != yplots - 1:
            axs[i % yplots, int(np.floor((i / yplots) % xplots))].set_xticklabels([])
        else:
            axs[i % yplots, int(np.floor((i / yplots) % xplots))].set_xticklabels([1988, 1998, 2008, 2018])

    axs[6, 0].set_ylabel('Percentage of documents containing term (%)', fontsize=12)

    # plt.show()
    plt.savefig('burst_example.png')
    plt.close(fig)

    with open('../models/burstiness_stacked_vec.pickle', 'wb') as handle:
        pickle.dump(stacked_vectors, handle)

#Build the development set
def build_development_set(stacked_vectors, unique_time_stamp, cutoff):
    reversed_stamp = deepcopy(unique_time_stamp)
    reversed_stamp.sort(reverse=True)

    print(reversed_stamp)
    recent = reversed_stamp[:10]
    print('--------------------------')
    print(recent)

    development_data = {}
    for _time_period in range(recent[len(recent)-1], recent[0]):
        _time_period_idx = unique_time_stamp.index(_time_period)

        # Use our three-year method to calc significance
        valid_vectors = calc_significance(stacked_vectors[:_time_period + 1], significance_threshold, 3)

        # Recalculate the macd things based on this more limited dataset
        long_ma, short_ma, significance_ma, macd, signal, hist = calc_macd(valid_vectors)

        #needs to be flexible for other types of timestamp - days, hours, minutes, etc
        # Calculate scaling factor
        scaling_factor = calc_scaling(significance_ma.iloc[max(long_ma_length, _time_period_idx - 13):_time_period_idx + 1], "sqrt")

        # Calculate the burstiness
        burstiness_over_time = calc_burstiness(hist, scaling_factor)
        burstiness = max_burstiness(burstiness_over_time)

        # Choose terms that are above both thresholds (burstiness, and also most recent year was significant)
        burst_idx = np.where((burstiness["max"] > 0.0012) & (significance_ma.iloc[_time_period_idx] > significance_threshold))[0]

        # Find the actual names of these terms
        bursts = valid_vectors.keys()[burst_idx]

        #needs to revist this logic -- minus 19! : update - 50 days?
        # Create a new, much smaller dataset

        dataset = stacked_vectors[bursts].iloc[_time_period_idx - 13:_time_period_idx + 1]

        # needs to set the threshold not just like 2015
        # Get the scaled y values
        if _time_period < cutoff:
            y = stacked_vectors[bursts].iloc[_time_period_idx + testing_period]

        # Select features and store the data
        development_data[_time_period] = {}
        development_data[_time_period]["X"] = feature_selection(dataset)
        if _time_period < cutoff:
            development_data[_time_period]["y"] = y - development_data[_time_period]["X"]["significance"]
        print(_time_period, len(bursts))

    print(len(development_data))
    return development_data

#Choosing a max depth for the random forest
def choose_max_depth(development_data, unique_time_stamp, cutoff):
    reversed_stamp = deepcopy(unique_time_stamp)
    reversed_stamp.sort(reverse=True)

    #
    # recent[len(recent)-3] is cutoff value
    #print(development_data)
    recent = reversed_stamp[:10]
    print(recent)

    X_array = []
    for year in range(recent[len(recent) - 3], cutoff):
        if development_data[year] is not None:
            X_array.append(development_data[year]["X"])
            print("FOUND!!!!!!!!!!!!!!!!!!!")

    if len(X_array) > 0:
        X = np.array(pd.concat(X_array))


    y_array = []
    for year in range(recent[len(recent) - 3], cutoff):
        if development_data[year] is not None:
            y_array.append(development_data[year]["y"])
            print("FOUND!!!!!!!!!!!!!!!!!!!")

    if len(y_array) > 0:
        y = np.array(pd.concat(y_array))
    else:
        return

    #X = np.array(pd.concat([development_data[year]["X"] for year in range(recent[len(recent)-3], cutoff)]))
    #y = np.array(pd.concat([development_data[year]["y"] for year in range(recent[len(recent)-3], cutoff)]))
    y_thresh = np.zeros_like(y)
    y_thresh[y > 0] = 1

    # Balance the sample
    X, y_thresh = balanced_subsample(X, y_thresh, subsample_size=1.0)
    x_train, x_test, y_train, y_test = train_test_split(X, y_thresh, test_size=0.33, random_state=1)

    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
        rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth)
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.title('Effect of changing the maximum depth of the random forest on classifier accuracy')
    plt.show()

#Choosing the number of estimators
def choose_estimators(development_data, unique_time_stamp):
    reversed_stamp = deepcopy(unique_time_stamp)
    reversed_stamp.sort(reverse=True)
    #
    # recent[len(recent)-3] is cutoff value
    #
    print(development_data)
    recent = reversed_stamp[:10]
    X = np.array(pd.concat([development_data[year]["X"] for year in range(recent[len(recent)-3], recent[0])]))
    y = np.array(pd.concat([development_data[year]["y"] for year in range(recent[len(recent)-3],recent[0])]))

    #X = np.array(pd.concat([development_data[year]["X"] for year in range(2008, 2015)]))
    #y = np.array(pd.concat([development_data[year]["y"] for year in range(2008, 2015)]))
    y_thresh = np.zeros_like(y)
    y_thresh[y > 0] = 1

    # Balance the sample
    X, y_thresh = balanced_subsample(X, y_thresh, subsample_size=1.0)
    x_train, x_test, y_train, y_test = train_test_split(X, y_thresh, test_size=0.33, random_state=3)

    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 150, 200, 300, 500, 800]
    train_results = []
    test_results = []
    for estimator in n_estimators:
        rf = RandomForestClassifier(n_estimators=estimator, max_depth=13, n_jobs=-1)
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
    line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.title('Effect of changing the number of estimators of the random forest on classifier accuracy')
    plt.show()

#The effect of changing the burstiness threshold and prediction interval on the random forest classifier
def change_bustiness_threshold(stacked_vectors, development_data):
    scores = {}
    for threshold in np.arange(0.0006, 0.0017, 0.0002):
        scores[threshold] = {}
        for year in range(2008, 2013):
            year_idx = year_range.index(year)

            # Use our three-year method to calc significance
            valid_vectors = calc_significance(stacked_vectors[:year_idx + 1], significance_threshold, 3)

            # Recalculate the macd things based on this more limited dataset
            long_ma, short_ma, significance_ma, macd, signal, hist = calc_macd(valid_vectors)

            # Calculate scaling factor
            scaling_factor = calc_scaling(significance_ma.iloc[max(long_ma_length, year_idx - 19):year_idx + 1], "sqrt")

            # Calculate the burstiness
            burstiness_over_time = calc_burstiness(hist, scaling_factor)
            burstiness = max_burstiness(burstiness_over_time)

            # Choose terms that are above both thresholds (burstiness, and also most recent year was significant)
            burst_idx = \
            np.where((burstiness["max"] > threshold) & (significance_ma.iloc[year_idx] > significance_threshold))[0]

            # Find the actual names of these terms
            bursts = valid_vectors.keys()[burst_idx]

            # Create a new, much smaller dataset
            dataset = stacked_vectors[bursts].iloc[year_idx - 19:year_idx + 1]

            # Select features and store the data
            development_data[year] = {}
            development_data[year]["X"] = feature_selection(dataset)

            development_data[year]["y"] = {}

            for interval in range(1, 6):
                # Get the scaled y values
                y = stacked_vectors[bursts].iloc[year_idx + interval]
                development_data[year]["y"][interval] = y - development_data[year]["X"]['significance']

        X = np.array(pd.concat([development_data[year]["X"] for year in range(2008, 2013)]))

        for interval in range(1, 6):
            scores[threshold][interval] = {}
            scores[threshold][interval]['scores'] = []
            y = np.array(pd.concat([development_data[year]["y"][interval] for year in range(2008, 2013)]))

            # Binarise y data
            y_thresh = np.zeros_like(y)
            y_thresh[y > 0] = 1

            # Balance the sample
            X_bal, y_thresh = balanced_subsample(X, y_thresh, subsample_size=1.0)

            scores[threshold][interval]['size'] = len(y_thresh)
            kf = KFold(n_splits=10, shuffle=True)
            for train, test in kf.split(X_bal):
                clf = RandomForestClassifier(n_estimators=150, max_depth=13)

                clf.fit(X_bal[train], y_thresh[train])
                preds = clf.predict(X_bal[test])

                new_scores = [
                    sklearn.metrics.accuracy_score(y_thresh[test], preds),
                    sklearn.metrics.f1_score(y_thresh[test], preds),
                    np.sum(y_thresh[test] == 0) / len(y_thresh[test])
                ]
                scores[threshold][interval]['scores'].append(new_scores)

            print(threshold, interval, len(y_thresh),
                  np.round(np.mean(np.array(scores[threshold][interval]['scores'])[:, 0]), 3)
                  )

#Format for table
def format_table(scores):
    for threshold in np.arange(0.0006, 0.0017, 0.0002):
        print(threshold, '&',
              scores[threshold][3]['size'], '&',
              np.round(np.mean(np.array(scores[threshold][3]['scores'])[:, 0]), 2),
              '$\pm$',
              np.round(np.std(np.array(scores[threshold][3]['scores'])[:, 0]), 2), '&',
              np.round(np.mean(np.array(scores[threshold][3]['scores'])[:, 1]), 2),
              '$\pm$',
              np.round(np.std(np.array(scores[threshold][3]['scores'])[:, 1]), 2),
              '\\\\'
              )
    plt.rc('font', family='sans-serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    line_styles = ['-', '--', ':']
    col = 0.5
    fig = plt.figure(figsize=(6, 3.7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Choosing a prediction interval, I', fontsize=13)
    ax.grid()
    ax.set_ylim(0.65, 0.9)
    # ax.set_xlim(1,5)

    ax.set_ylabel('F1 score', fontsize=12)
    ax.set_xlabel('Number of years in future', fontsize=12)

    plt.xticks(range(1, 6), range(1, 6))

    thresholds = ["0.0006", "0.0008", "0.0010", "0.0012", "0.0014", "0.0016"]
    for i, threshold in enumerate(np.arange(0.0006, 0.0017, 0.0002)):
        y = []
        yerr = []
        for interval in range(1, 6):
            s = np.array(scores[threshold][interval]['scores'])
            y.append(np.mean(s[:, 1]))
            yerr.append(np.std(s[:, 1]))

        ax.errorbar(range(1, 6), y, yerr=yerr, color=str(col), label=thresholds[i], fmt='--o')

        col -= 0.1

    # ax.legend(ncol=3, mode="expand")
    ax.legend(fontsize=11)

#Train a single classifier on all the data from 1988-2014
def train_classifier(stacked_vectors, unique_time_stamp, cutoff):
    threshold = 0.0012
    interval = 3

    reversed_stamp = deepcopy(unique_time_stamp)
    reversed_stamp.sort(reverse=True)

    print(reversed_stamp)
    recent = reversed_stamp[:10]
    print('--------------------------')
    print(recent)

    development_data = {}
    for _time_period in range(recent[len(recent) - 1], recent[0]):
        _time_period_idx = unique_time_stamp.index(_time_period)
        # Use our three-year method to calc significance
        valid_vectors = calc_significance(stacked_vectors[:_time_period_idx + 1], significance_threshold, 3)

        # Recalculate the macd things based on this more limited dataset
        long_ma, short_ma, significance_ma, macd, signal, hist = calc_macd(valid_vectors)

        # Calculate scaling factor
        scaling_factor = calc_scaling(significance_ma.iloc[max(long_ma_length, _time_period_idx - 13):_time_period_idx + 1], "sqrt")

        # Calculate the burstiness
        burstiness_over_time = calc_burstiness(hist, scaling_factor)
        burstiness = max_burstiness(burstiness_over_time)

        # Choose terms that are above both thresholds (burstiness, and also most recent year was significant)
        burst_idx = \
        np.where((burstiness["max"] > threshold) & (significance_ma.iloc[_time_period_idx] > significance_threshold))[0]

        # Find the actual names of these terms
        bursts = valid_vectors.keys()[burst_idx]

        # Create a new, much smaller dataset
        dataset = stacked_vectors[bursts].iloc[_time_period_idx - 13:_time_period_idx + 1]

        # Select features and store the data
        development_data[_time_period] = {}
        development_data[_time_period]["X"] = feature_selection(dataset)
        if _time_period < cutoff:
            y = stacked_vectors[bursts].iloc[_time_period + interval]
            development_data[_time_period]["y"] = y - development_data[_time_period]["X"]['significance']

    recent = reversed_stamp[:10]
    print(recent)

    X_array = []
    for year in range(recent[len(recent) - 3], cutoff):
        if development_data[year] is not None:
            X_array.append(development_data[year]["X"])
            print("FOUND!!!!!!!!!!!!!!!!!!!")

    if len(X_array) > 0:
        X = np.array(pd.concat(X_array))

    y_array = []
    for year in range(recent[len(recent) - 3], cutoff):
        if development_data[year] is not None:
            y_array.append(development_data[year]["y"])
            print("FOUND!!!!!!!!!!!!!!!!!!!")

    if len(y_array) > 0:
        y = np.array(pd.concat(y_array))
    else:
        return

    #X = np.array(pd.concat([development_data[year]["X"] for year in range(2008,2015)]))
    #y = np.array(pd.concat([development_data[year]["y"] for year in range(2008,2015)]))

    # Binarise y data
    y_thresh = np.zeros_like(y)
    y_thresh[y>0] = 1

    # Balance the sample
    X_bal, y_thresh = balanced_subsample(X, y_thresh,subsample_size=1.0)

    clf = RandomForestClassifier(n_estimators=150, max_depth=13)
    clf.fit(X_bal, y_thresh)

    return clf

def predict(burstiness, significance_ma, valid_vectors, stacked_vectors, year_idx, clf):
    # Choose terms that are above both thresholds
    burst_idx = np.where(
        (burstiness["max"] > burstiness_threshold_prediction) & (significance_ma.iloc[29] > significance_threshold))[0]

    # Find the actual names of these terms
    bursts = list(valid_vectors.keys()[burst_idx])

    # Create a new, much smaller dataset
    dataset = stacked_vectors[bursts].iloc[year_idx - 19:year_idx + 1]

    X_test = np.array(feature_selection(dataset))

    preds = clf.predict(X_test)
    rise = []
    fall = []
    for i, p in enumerate(preds):
        if p > 0.5:
            rise.append(bursts[i])
        else:
            fall.append(bursts[i])

    print(', '.join(rise))
    print()
    print(', '.join(fall))


def compute_pure_term_burstiness(documents, label_list):
    # Build a vocabulary
    # We have to build a vocabulary before we vectorise the data. This is because we want to set limits on the size of the vocabulary.

    vocab = set()

    df = pd.DataFrame([(label_list, documents) for label_list, documents in zip(label_list, documents)])
    print(df)

    grouped_df = df.groupby(df.columns[0])

    for key, item in grouped_df:
        a_group = grouped_df.get_group(key)

        print(a_group.head(10))
        # The same as above, applied year by year instead.
        t0 = time.time()

        vectorizer = CountVectorizer(min_df=min_yearly_df)
        print(a_group.iloc[:,1].head(10))

        vector = vectorizer.fit_transform(a_group.iloc[:, 1])

        # Save the new words
        vocab = vocab.union(vectorizer.vocabulary_.keys())
        time_stamp = a_group.iloc[:, 0].tolist()[0]
        print(time_stamp, len(vocab), time.time() - t0)

    vocabulary = {}
    i = 0
    for v in vocab:
        vocabulary[v] = i
        i += 1

    print(len(vocabulary.keys()))

    vocab = pd.DataFrame(vocabulary.items())[[0]]
    #vocab.to_csv('../sample_data/vocab.csv')
    # Go year by year and vectorise based on our vocabulary
    # We read in the cleaned data and vectorise it according to our vocabulary.
    vectors = []
    grouped_df = df.groupby(df.columns[0])

    for key, item in grouped_df:
        a_group = grouped_df.get_group(key)

        # The same as above, applied year by year instead.
        t0 = time.time()

        vectorizer = CountVectorizer(vocabulary=vocabulary)

        vectors.append(vectorizer.fit_transform(a_group.iloc[:, 1]))
        time_stamp = a_group.iloc[:, 0].tolist()[0]
        print(time_stamp, time.time() - t0)

    # Summing the vectors
    # We sum the vectors along columns, so that we have the popularity of each term in each year.
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
    #print(stacked_vectors.shape)

    stacked_vectors = pd.DataFrame(stacked_vectors.transpose(), columns=list(vocabulary.keys()))
    #print(stacked_vectors.shape)

    #stacked_vectors.to_csv('../sample_data/stacked_vectors.csv')
    #stacked_vectors.index = list(range(20170101, 20171024))
    print(stacked_vectors.index)

    long_ma, short_ma, significance_ma, macd, signal, hist = calc_macd(stacked_vectors)
    scaling_factor = calc_scaling(significance_ma, "mean")
    #print(hist)
    burstiness_over_time = calc_burstiness(hist, scaling_factor)
    #burstiness = max_burstiness(burstiness_over_time)

    dir = "./results"
    if not os.path.exists(dir):
        os.makedirs(dir)

    burstiness_over_time.to_csv(dir + '/bursty_terms.csv')

    print(burstiness_over_time)
