
import requests
import random
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver

class NaverBlog():
    def __init__(self):
        self.ab_url=[]
        self.query = ''

    def get_link(self, query, s_date, e_date):
        self.query = query
        self.s_date=s_date
        self.e_date=e_date
        options = webdriver.ChromeOptions()
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36")

        options.add_argument('headless')
        options.add_argument('--disable-gpu')
        options.add_argument('lang=ko_KR')
        browser = WebDriver(executable_path='D:\python_workspace\pyTextMiner\selenium_server\chromedriver.exe', options=options)
        #browser = WebDriver(executable_path='/usr/lib/chromium-browser/chromedriver', options=options)
        url = "https://m.search.naver.com/search.naver?where=m_blog&sm=mtb_opt&query=" + query + "&display=15&st=sim&nso=p%3Afrom" + s_date + "to" + e_date
        browser.get(url)
        browser.implicitly_wait(random.randrange(5,10))
        SCROLL_PAUSE_TIME = 1.5
        # Get scroll height
        last_height = browser.execute_script("return document.body.scrollHeight")
        while True:
            # Scroll down to bottom
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)
            # Calculate new scroll height and compare with last scroll height
            new_height = browser.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                cont = browser.page_source
                soup = BeautifulSoup(cont, 'html.parser')
                for urls in soup.select(".total_dsc"):
                    if urls["href"].startswith("https://m.blog.naver.com") or 'blog.me' in urls["href"]:
                        self.ab_url.append(urls['href'])
                break
            last_height = new_height
        time.sleep(random.randrange(5,15))

    def get_detail(self):
        f = open(self.query + "_" + self.s_date + "_" + self.e_date + '.txt', 'w', encoding='utf-8')
        for url in self.ab_url:
            print(url)
            time.sleep(random.randrange(10, 15))
            header = {
                'User-Agent':'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36',
            }
            breq = requests.get(url, headers=header)
            time.sleep(random.randrange(20,30))
            cont = breq.content
            bsoup = BeautifulSoup(cont, 'html.parser')
            title = bsoup.select('.se-module.se-module-text.se-title-text')
            if title ==[]:
                title=bsoup.select('.tit_h3')
                if title == []:
                    title = bsoup.select('.se_textarea')
                    title2 = title[0].text.strip().replace('\n', " ").replace("\r", " ")
                    f.write(title2 + '\t')
                    print(title2)
                title2=title[0].text.strip().replace('\n', " ").replace("\r", " ")
                f.write(title2+'\t')
                print(title2)
            else:
                title2 = title[0].text.strip().replace('\n', " ").replace("\r", " ")
                f.write(title2+'\t')
                print(title2)
            time.sleep(3)

            pdate = bsoup.select('.blog_date')
            if pdate==[]:
                pdate=bsoup.select('.se_date')
                pdate2=pdate[0].get_text()
                f.write(pdate2.strip()+'\t')
                print(pdate2)
            else:
                pdate2=pdate[0].get_text()
                f.write(pdate2.strip()+'\t')
                print(pdate2)

            user = bsoup.select('.blog_author')
            if user == []:
                user = bsoup.select('.se_author')
                user2 = user[0].a.get_text()
                f.write(user2.strip()+'\t')
                print(user2)
            else:
                user2 = user[0].a.get_text()
                f.write(user2.strip() + '\t')
                print(user2)

            _text = bsoup.select('.se-main-container')
            if _text==[]:
                _text=bsoup.select('.post_ct')
                if _text == []:
                    _text = bsoup.select('.se_textarea')
                    text2 = _text[0].get_text().replace('\n', " ").replace("\r", " ").replace("\t", " ")
                    f.write(text2 + '\t')
                    print(text2.strip())
                text2 = _text[0].get_text().replace('\n', " ").replace("\r", " ").replace("\t", " ")
                f.write(text2+'\t')
                print(text2.strip())
            else:
                text2=_text[0].get_text().replace('\n', " ").replace("\r", " ").replace("\t", " ")
                f.write(text2+'\t')
                print(text2.strip())

            f.write(url+'\n')
        f.close()

if __name__ == '__main__':
    query = '우울증'
    s_date = "2017.01.01"
    e_date = "2017.01.31"

    s_from = s_date.replace(".", "")
    e_to = e_date.replace(".", "")

    blog = NaverBlog()
    blog.get_link(query,s_from,e_to)
    blog.get_detail()