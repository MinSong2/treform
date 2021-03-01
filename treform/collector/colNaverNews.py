from treform.collector.colBasic import *

class ColNaverNews(ColBasic):
    def getInfo(params):
        return '네이버 뉴스 수집기', 0

    def getName(self):
        return "colNaverNews"

    def getSearchURL(self):
        return 'https://m.search.naver.com/search.naver'

    def makeParameter(self, page, query, d):
        if page*16 >= 4000: raise Exception("Cannot collect over 4000 articles")
        return {'where':'m_news',
                'query':query,
                'sm':'mtb_tnw',
                'sort':1,
                'pd':3,
                'ds':self.reformatDate(d['startDate'], '%04d.%02d.%02d'),
                'de':self.reformatDate(d['endDate'], '%04d.%02d.%02d'),
                'start':page*16 - 15,
            }

    def selectList(self, soup):
        ret = []
        for el in soup.select('.list_news a'):
            link = el['href']
            if link.find('news.naver.com') < 0: continue
            ret.append(link)
        return ret

    def selectArticle(self, soup):
        [x.extract() for x in soup.findAll('script')]
        return (ColNaverNews.cleanText(soup.select_one('.media_end_head_info_datestamp_time').text),
                ColNaverNews.cleanText(soup.select_one('h2').text),
                ColNaverNews.cleanText(soup.select_one('.newsct_article').text))
