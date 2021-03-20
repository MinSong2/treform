import requests
from bs4 import BeautifulSoup

url = 'http://finance.naver.com/'
res = requests.get(url)
text = res.text
#print(text)

soup = BeautifulSoup(text, 'html.parser')
#print(soup)

td = soup.select_one("#content > div.article2 > div.section1 > div.group1 > table > tbody > tr > td")
print(td.text)