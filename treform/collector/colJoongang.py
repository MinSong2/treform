from treform.collector.colBasic import *

class ColJoongang(ColBasic):
    def getInfo(params):
        return '중앙일보 수집기', 0

    def getName(self):
        return "colJoongang"

    def getSearchURL(self):
        return 'http://search.joins.com/JoongangNews'

    def makeParameter(self, page, query, d):
        return {'StartSearchDate': d['startDate'],
                'EndSearchDate': d['endDate'],
                'Keyword': query,
                'SortType': 'New',
                'SearchCategoryType': 'JoongangNews',
                'PeriodType': 'DirectInput',
                'ScopeType': 'All',
                'ServiceCode': '',
                'SourceGroupType': '',
                'ReporterCode': '',
                'ImageType': 'All',
                'JplusType': 'All',
                'BlogType': 'All',
                'ImageSearchType': 'Image',
                'MatchKeyword': '',
                'IncludeKeyword': '',
                'ExcluedeKeyword': '',
                'page':page,
        }

    def selectList(self, soup):
        ret = []
        for el in soup.select('.headline a'):
            link = el['href']
            ret.append(link)
        return ret

    def selectArticle(self, soup):
        return (ColBasic.getTextFromElement(soup.select_one('h1')),
                ColBasic.cleanText(soup.find('meta', {'property': 'article:published_time'})['content']),
                ColBasic.getTextFromElement(soup.select_one('.article_body')))
