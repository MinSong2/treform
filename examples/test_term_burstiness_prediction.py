import pickle
from treform.weighting.term_burstiness import build_development_set, choose_max_depth, \
    choose_estimators, train_classifier

with open('../models/burstiness_stacked_vec.pickle', 'rb') as handle:
    stacked_vectors = pickle.load(handle)

with open('../models/unique_time_stamp.pickle', 'rb') as handle:
    unique_time_stamp = pickle.load(handle)

cutoff = 20170117
development_data = build_development_set(stacked_vectors, unique_time_stamp, cutoff)
choose_max_depth(development_data,unique_time_stamp,cutoff)
#choose_estimators(development_data,unique_time_stamp)
classifier = train_classifier(stacked_vectors, unique_time_stamp, cutoff)

from datetime import datetime, timedelta
from dateutil import parser

DAY = timedelta(days=50)
print(DAY)
now = datetime.now() # get timezone-aware datetime object
print(now)

date1 = datetime.strptime(str(cutoff), '%Y%m%d')
print(date1)

naive = date1 - DAY # same time

f_naive = parser.parse(str(naive)).strftime('%Y%m%d')
print(f_naive)

#20170124, 20170123, 20170122, 20170121, 20170120, 20170119, 20170118, 20170117

year_range = list(range(20170101,20170124))
for year in range(20170115, 20170124):
    print(year)
    print("------------------------")
    year_idx = year_range.index(year)
    print (year_idx)
    print ('-----------------')
    print (str(year_idx - 13))
    print('-----------------')
    print (str(year_idx + 1))
    print()
    print()