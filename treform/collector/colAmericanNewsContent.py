import newspaper3k
import os
import time
import random
import sys
import re

reSpace = re.compile('\\s+')
metaDatePublished = re.compile('<meta *itemprop="datePublished" *content="([^"]+)" */?>')
metaUploadDate = re.compile('<meta *itemprop="uploadDate" *content="([^"]+)" */?>')
metaUtime = re.compile('<meta *name="utime" *content="([0-9]{4})([0-9]{2})([0-9]{2}).*?" */?>')
metaPdate = re.compile('<meta *name="pdate" *content="([0-9]{4})([0-9]{2})([0-9]{2}).*?" */?>')
rawDateText = re.compile('(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[^\\s]* *([0-9]{1,2}), *([0-9]{4})')

def extractDate(html):
    m = metaDatePublished.search(html)
    if m: return m.group(1)
    m = metaUploadDate.search(html)
    if m: return m.group(1)
    m = metaUtime.search(html)
    if m: return m.group(1) + '-' + m.group(2) + '-' + m.group(3)
    m = metaPdate.search(html)
    if m: return m.group(1) + '-' + m.group(2) + '-' + m.group(3)
    m = rawDateText.search(html)
    if m: return m.group(0)
    return None

def getArticle(url):
    import dateutil.parser
    article = newspaper.Article(url, keep_article_html=True)
    try:
        article.download()
        article.parse()
    except Exception as e:
        raise e
    date = article.publish_date or dateutil.parser.parse(extractDate(article.html)).strftime('%Y-%m-%d %H:%M:%S')
    if not date: raise Exception("Cannot find date")
    return (url, article.title, ','.join(article.authors), str(date), article.text.replace('\n', ' '))

if __name__ == "__main__":
    keyword = sys.argv[1]
    output = open('data/contents/%s.txt' % keyword, 'w', encoding='utf-8')
    for name in os.listdir('data/url'):
        if name.find(keyword) < 0: continue
        for url in set(open('data/url/' + name, encoding='utf-8').readlines()):
            print("Download %s ..." % url.strip())
            try:
                output.write("\t".join(getArticle(url.strip())) + "\n")
            except:
                print("Failed to fetch page : %s"  % url.strip())
            time.sleep(random.randint(4, 10))
    output.close()