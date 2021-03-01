class ColBasic:
    def __init__(self):
        self.userAgent = 'Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0'
        self.sleepA = 9
        self.sleepB = 3

    def getInfo(params):
        return '기본 수집기', 0

    def getName(self):
        return "colBasic"

    def parseDate(self, date):
        import re
        m = re.match('([1-2][0-9]{3})([-.]?)([0-2][0-9])\\2([0-3][0-9])', date)
        if m: return (int(m.group(1)), int(m.group(3)), int(m.group(4)))
        return None

    def reformatDate(self, date, format):
        d = self.parseDate(date)
        if not d: return ''
        return format % d

    def initParameter(self, params):
        startDate = params['startDate']
        endDate = params['endDate']
        if params.get('lastDate'): endDate = params['lastDate']
        startDate = self.reformatDate(startDate, '%04d.%02d.%02d')
        endDate = self.reformatDate(endDate, '%04d.%02d.%02d')
        return {'startDate':startDate, 'endDate':endDate}

    def fetchList(self, page, query, d):
        import urllib.parse, urllib.request, bs4
        data = urllib.parse.urlencode(self.makeParameter(page, query, d))
        req = urllib.request.Request(self.getSearchURL() + '?' + data)
        req.add_header('Referer', self.getSearchURL())
        req.add_header('User-Agent', self.userAgent)
        f = urllib.request.urlopen(req)
        cont = f.read().decode('utf-8')
        soup = bs4.BeautifulSoup(cont, "lxml")
        return self.selectList(soup)

    @staticmethod
    def cleanText(t):
        return t.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

    def fetchArticle(self, id):
        import urllib.request, bs4, traceback, time, random
        try:
            req = urllib.request.Request(id)
            req.add_header('Referer', self.getSearchURL())
            req.add_header('User-Agent', self.userAgent)
            f = urllib.request.urlopen(req)
            bytestream = f.read()
            try:
                cont = bytestream.decode('utf-8')
            except:
                cont = bytestream.decode('euc-kr')
            soup = bs4.BeautifulSoup(cont, "lxml")
            return (id, ) + self.selectArticle(soup)
        except:
            traceback.print_exc()
            time.sleep(random.random() * self.sleepA + self.sleepB + 1)

    def collect(self, query, outputPath, **params):
        import sys, traceback
        import time, random
        try:
            page = int(params['page'])
        except:
            page = 1
        d = self.initParameter(params)
        output = open(outputPath, 'w', encoding='utf-8')
        try:
            while 1:
                urls = self.fetchList(page, query, d)
                fetched = 0
                for url in urls:
                    article = self.fetchArticle(url)
                    if not article: continue
                    fetched += 1
                    print(article)
                    output.write('\t'.join(map(str, article)) + '\n')
                    output.flush()
                    time.sleep(random.random() * self.sleepA + self.sleepB)
                if fetched == 0 and len(urls) > 5:
                    raise RuntimeError('Failed to fetch articles \n' + '\n'.join(urls))
                page += 1
        except:
            traceback.print_exc()
        output.close()

    def getSearchURL(self):
        pass

    def makeParameter(self, page, query, d):
        pass

    def selectList(self, soup):
        pass

    def selectArticle(self, soup):
        pass

    @staticmethod
    def getTextFromElement(soup):
        if not soup: return None
        [s.extract() for s in soup('script')]
        [s.extract() for s in soup('style')]
        return ColBasic.cleanText(soup.text)