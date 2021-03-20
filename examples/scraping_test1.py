import requests
from bs4 import BeautifulSoup

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko'
}
url = 'https://www.acmicpc.net/ranklist'
res = requests.get(url, headers=headers)
text = res.text
soup = BeautifulSoup(text,'html.parser')

for tr in soup.select('#ranklist > tbody > tr'):
    #print(tr)
    tds = tr.select('td')
    rank = tds[0].text.strip()
    id = tds[1].text.strip()
    message = tds[2].text.strip()
    solved = tds[3].text.strip()
    submitted = tds[4].text.strip()
    accuracy = tds[5].text.strip()
    print(rank,id,message,solved,submitted,accuracy)
    #print('----------')