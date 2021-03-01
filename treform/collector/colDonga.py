from treform.collector.colBasic import *

class ColDonga(ColBasic):
    def getInfo(params):
        return '동아일보 수집기', 0

    def getName(self):
        return "colDonga"

    def getSearchURL(self):
        return 'http://news.donga.com/search'

    def initParameter(self, params):
        startDate = params['startDate'][0]
        endDate = params['endDate'][0]
        if params.get('lastDate'): endDate = params['lastDate']
        startDate = self.reformatDate(startDate, '%04d%02d%02d')
        endDate = self.reformatDate(endDate, '%04d%02d%02d')
        return {'startDate':startDate, 'endDate':endDate}

    def makeParameter(self, page, query, d):
        return {'p':page*15 - 14,
            'query':query,
            'check_news':'1',
            'more':'1',
            'sorting':'1',
            'search_date':'5',
            'v1':d['startDate'],
            'v2':d['endDate'],
            'range':'1',
        }

    def selectList(self, soup):
        ret = []
        for el in soup.select('.txt a'):
            link = el['href']
            ret.append(link)
        return ret

    def selectArticle(self, soup):
        return (ColBasic.getTextFromElement(soup.select_one('h2.title')),
                ColBasic.cleanText(soup.find('meta', {'property': 'article:published_time'})['content']),
                ColBasic.getTextFromElement(soup.select_one('.article_txt')))
