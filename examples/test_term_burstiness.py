import treform as ptm

from treform.weighting.term_burstiness import compute_term_burstiness

dataset = '../sample_data/news_articles_201701_201812.csv'
compute_term_burstiness(dataset)