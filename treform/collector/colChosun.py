from treform.collector.colBasic import *

class ColChosun(ColBasic):
    def getInfo(params):
        return '조선일보 수집기', 0

    def getName(self):
        return "colChosun"

    def getSearchURL(self):
        return 'https://nsearch.chosun.com/search/total.search'

    def makeParameter(self, page, query, d):
        return {'query':query,
                    'pageno':page,
                    'orderby':'docdatetime',
                    'naviarraystr':'',
                    'kind':'',
                    'cont1':'',
                    'cont2':'',
                    'cont5':'',
                    'categoryname':'',
                    'categoryd2':'',
                    'c_scope':'paging',
                    'sdate':d['startDate'],
                    'edate':d['endDate'],
                    'premium':'',
            }

    def selectList(self, soup):
        ret = []
        for el in soup.select('.search_news dt a'):
            link = el['href']
            ret.append(link)
        return ret

    def selectArticle(self, soup):
        return (ColBasic.getTextFromElement(soup.select_one('h1')),
                ColBasic.cleanText(soup.select_one('#date_text').text.replace('입력 :', '').strip()[:17].strip()) if soup.select_one('#date_text') else None,
                ColBasic.getTextFromElement(soup.select_one('.par')))
