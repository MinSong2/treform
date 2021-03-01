# -*- coding: utf-8 -*-
import datetime

from selenium import webdriver
import time
import random
from urllib.parse import quote_plus

class NewspaperCollector:
    def __init__(self, **args):
        self.wait_min = args.get('wait_min', 1)
        self.wait_max = args.get('wait_max', 10)
        self.wait_loading = args.get('wait_loading', 60)
        self.browser = None
        self.keyword = None

    def wait(self):
        time.sleep(random.randint(self.wait_min, self.wait_max))

    def collectByKeyword(self, keyword, output, **args):
        self.keyword = keyword
        article_list = set()
        if output:
            outputFile = open(output, 'w')
        maxNum = args.get('maxNum', -1)
        print('Init Chromedriver...')
        #self.browser = webdriver.PhantomJS(executable_path=r"D:\python_workspace\pyTextMiner\selenium_server\phantomjs.exe")
        self.browser=webdriver.Chrome(executable_path='D:\python_workspace\pyTextMiner\selenium_server\chromedriver.exe')
        url = self.getFirstURL(keyword, **args)
        print('Get Page... ' + url)

        self.browser.get(url)
        self.wait()
        curPage = 0
        while 1:
            secs = 0
            while secs < self.wait_loading:
                links = self.getLinkers(self.browser)
                if len(links): break
                print("Wait for page loading...")
                time.sleep(5)
                secs += 5
            if secs >= self.wait_loading: break
            oLen = len(article_list)
            for l in links:
                href = l.get_attribute('href')
                if href not in article_list and outputFile: outputFile.write(href + "\n")
                article_list.add(href)
                print('url ' + url)
                if maxNum > 0 and len(article_list) >= maxNum: break
            if oLen == len(article_list): break
            if maxNum > 0 and len(article_list) >= maxNum: break
            if outputFile: outputFile.flush()
            try:
                curPage += 1
                print(self.browser.current_url)
                self.getNext(self.browser, curPage)
                print('Get Next Page...')
                self.wait()
            except Exception as e:
                print(e)
                break
        self.browser.quit()
        self.browser = None
        if outputFile: outputFile.close()
        return article_list

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.browser: self.browser.quit()
        self.browser = None

class CNNCollector(NewspaperCollector):
    def getFirstURL(self, keyword, **args):
        return "https://edition.cnn.com/search/?q=" + quote_plus(keyword)

    def getLinkers(self, browser):
        return browser.find_elements_by_css_selector('h3.cnn-search__result-headline a')

    def getNext(self, browser, nextPage):
        browser.find_element_by_css_selector('.pagination-arrow-right').click()

class NYTCollector(NewspaperCollector):
    def getFirstURL(self, keyword, **args):
        #base = "https://query.nytimes.com/search/sitesearch/?action=click&contentCollection&region=TopBar&WT.nav=searchWidget&module=SearchSubmit&pgtype=Homepage#/"
        base="https://www.nytimes.com/search?query="
        time = args.get('time', 'since1851')
        if time == 'since1851':
            time = 'startDate=18510101'
        elif time == '24hours':
            x = datetime.datetime.today()
            a_x = datetime.datetime.today() - datetime.timedelta(days=1)
            time = 'endDate='+str(x.year) + str(x.month) + str(x.day) + '&startDate=' +str(a_x.year) + str(a_x.month) + str(a_x.day)
        elif time == '7days':
            x = datetime.datetime.today()
            a_x = datetime.datetime.today() - datetime.timedelta(days=7)
            time = 'endDate='+str(x.year) + str(x.month) + str(x.day) + '&startDate=' +str(a_x.year) + str(a_x.month) + str(a_x.day)
        elif time == '30days':
            x = datetime.datetime.today()
            a_x = datetime.datetime.today() - datetime.timedelta(days=30)
            time = 'endDate='+str(x.year) + str(x.month) + str(x.day) + '&startDate=' +str(a_x.year) + str(a_x.month) + str(a_x.day)
        elif time == '12months':
            x = datetime.datetime.today()
            a_x = datetime.datetime.today() - datetime.timedelta(months=12)
            time = 'endDate='+str(x.year) + str(x.month) + str(x.day) + '&startDate=' +str(a_x.year) + str(a_x.month) + str(a_x.day)

        result_type = args.get('resultType', 'all')
        if (result_type == "all"):
            return base + quote_plus(keyword) + "&" + time
        else:
            return base + quote_plus(keyword) + "/" + time + "&type=article"

    def getLinkers(self, browser):
        #li.css-1l4w6pd
        return browser.find_elements_by_css_selector('li.css-1l4w6pd div a')

    def getNext(self, browser, nextPage):
        #data-testid="search-show-more-button
        browser.find_element_by_css_selector('[data-testid="search-show-more-button"]').click()

class WPCollector(NewspaperCollector):
    def getFirstURL(self, keyword, **args):
        return "https://www.washingtonpost.com/newssearch/?query=" + quote_plus(keyword) + "#page-1"

    def getLinkers(self, browser):
        return browser.find_elements_by_css_selector('.pb-feed-item a.ng-binding')

    def getNext(self, browser, nextPage):
        browser.get("https://www.washingtonpost.com/newssearch/?query=" + quote_plus(self.keyword) + ("#page-%d" % (nextPage+1)))

class WSJCollector(NewspaperCollector):
    def getFirstURL(self,keyword,**args):
        from datetime import date
        today = date.today()
        minDate = "%04d/%02d/%02d" % (today.year-4, today.month, today.day)
        maxDate = "%04d/%02d/%02d" % (today.year, today.month, today.day)
        return "https://www.wsj.com/search/term.html?KEYWORDS="+quote_plus(keyword)  +"&min-date="+minDate+"&max-date="+maxDate+"&daysback=4y&isAdvanced=true&andor=AND&sort=date-desc&source=wsjarticle,wsjblogs,wsjvideo,sitesearch"


    def getLinkers(self, browser):
        return browser.find_elements_by_css_selector('h3.headline a')

    def getNext(self, browser, nextPage):
        browser.find_element_by_css_selector('.next-page a').click()

class ForbesCollector(NewspaperCollector):
    def collectByKeyword(self, keyword, output, **args):
        import json
        import urllib.request
        self.keyword = keyword
        article_list = []
        outputFile = None
        if output:
            outputFile = open(output, 'w')
        maxNum = args.get('maxNum', -1)
        curPage = 0
        while 1:
            response = urllib.request.urlopen('https://www.forbes.com/forbesapi/search/all.json?limit=100&orfilters=&query=%s&retrievedfields=author,authors,blogType,date,description,editorsPick,hashtags,naturalId,trendingHashtags,image,slides,title,type,uri&sort=score&start=%d&startdate=&withentity=false' % (quote_plus(keyword), curPage*100))
            d = json.loads(response.read().decode('utf-8'))
            if not 'contentList' in d: break
            for l in d['contentList']:
                if not 'uri' in l: continue
                href = l['uri']
                if not href.startswith('http'): continue
                #print(href)
                article_list.append(href)
                if outputFile: outputFile.write(href + "\n")
                if maxNum > 0 and len(article_list) >= maxNum: break
            if maxNum > 0 and len(article_list) >= maxNum: break
            if outputFile: outputFile.flush()
            curPage += 1
            self.wait()
        if outputFile: outputFile.close()
        return article_list


allClass = [NYTCollector, CNNCollector, WPCollector, ForbesCollector, WSJCollector]

if __name__ == "__main__":
    import sys
    import os

    time_range_list = ["since1851", "24hours", "7days", "30days", "12months"]
    result_type_list = ["all", "article", "blogpost", "topic", "multimedia"]
    site_type_list = ['All', 'New York Times', 'CNN', 'Washington Post', 'Forbes', 'WSJ']
    site_class = allClass

    if (len(sys.argv) > 3):
        term_1 = str(sys.argv[1])
        term_2 = str(sys.argv[2])
        site = int(sys.argv[3])
        num = int(sys.argv[4])
        _time = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        _type = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        term = term_1+" "+term_2

        if not os.path.exists("data/url"): os.makedirs("data/url")
        if site == 0:
            for n, cls in enumerate(site_class):
                print("Collecting article related to '%s' from %s" % (term_1 + " AND " +term_2, site_type_list[n+1]))
                with cls(wait_min = 5, wait_max = 10) as inst:
                    urls = inst.collectByKeyword(term, "data/url/%s_%s.txt" % (term_1 + "_" +term_2, site_type_list[n+1]), maxNum = num, time=time_range_list[_time], resultType=result_type_list[_type])
                    print("%d urls are collected." % len(urls))
        else:
            print("Collecting article related '%s' from %s" % (term_1 + " AND " +term_2, site_type_list[site]))
            with site_class[site-1](wait_min=5, wait_max=10) as inst:
                urls = inst.collectByKeyword(term, "data/url/%s_%s.txt" % (term_1 + "_" +term_2, site_type_list[site]), maxNum = num,
                                  time=time_range_list[_time], resultType=result_type_list[_type])
                print("%d urls are collected." % len(urls))
    else:
        print(
            "============================================="+
            "\n==== Extract Url From Online Newspaper =====\n\nPlease try again with 3~5 arguments :(1)Keywords (2)Newspaper Site (3)Number of articles [(4)time_range (5)Result_type]")
        print("Ex> extract_url.py Informatics apple 0 100 0 0\n")
        print("To crawl all articles, set 'Number of articles' to -1")

        print("-----------------Site Type -----------------")
        for i, each in enumerate(site_type_list):
            print(each + " : " + str(i))
        print("=========================================\n")
        print("-----------------Time range -----------------")
        for i, each in enumerate(time_range_list):
            print(each + " : " + str(i))
        print("=========================================\n")
        print("-----------------Result type-----------------")
        for i, each in enumerate(result_type_list):
            print(each + " : " + str(i))
        print("=========================================\n")
