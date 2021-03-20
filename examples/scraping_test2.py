from selenium import webdriver
import time
from bs4 import BeautifulSoup

wd = webdriver.Chrome(r'D:\python_workspace\treform\selenium_server\chromedriver.exe')

#'''
url = 'https://www.goobne.co.kr/store/search_store.jsp'
wd.implicitly_wait(3)
wd.get(url)
select = wd.find_element_by_css_selector('#sSelsi > option:nth-child(12)')
select.click()
a = wd.find_element_by_css_selector('#contents > form > div > div > ul > li:nth-child(1) > p.area_search > a')
a.click()

time.sleep(3)
text = wd.page_source
#print(text)

soup = BeautifulSoup(text,'html.parser')
#print(soup)

for tr in soup.select('#store_list > tr'):
    branch = tr.select_one('td').text
    store_phone = tr.select_one('td.store_phone').text
    address = tr.select_one('td.t_left > a').text
    print(branch,store_phone,address)
#'''
'''
url = 'https://www.goobne.co.kr/store/search_store.jsp'
wd.implicitly_wait(3) #초
wd.get(url)
select = wd.find_element_by_css_selector('#sSelsi > option:nth-child(12)')
select.click()
a = wd.find_element_by_css_selector('#contents > form > div > div > ul > li:nth-child(1) > p.area_search > a')
a.click()

for tr in wd.find_elements_by_css_selector('#store_list > tr.lows'): #
    branch = tr.find_element_by_css_selector('td').text
    store_phone = tr.find_element_by_css_selector('td.store_phone').text   
    address = tr.find_element_by_css_selector('td.t_left > a').text
    print(branch,store_phone,address)
'''

wd.quit()

'''
현풍점 053-617-9492 대구광역시 달성군 현풍면 현풍중앙로14길 85
장기감삼점 053-523-9492 대구광역시 달서구 장기동 528
유가점 053-616-9482 대구광역시 달성군 유가면 봉리 592-1
신월성점 053-639-9944 대구광역시 달서구 월배로 33길 27 (진천동)
성서계명대점 053-584-9288 대구광역시 달서구  서당로9안길 5
성당본리점 053-628-9494 대구광역시 달서구  대명천로 96
불로봉무점 053-985-9294 대구광역시 동구 팔공로101길 47
본동월성2동점 053-521-9492 대구광역시 달서구 본동 710-1
범어1,2,3동점 053-751-9492 대구광역시 수성구  동대구로73길 24
대명1호점 053-621-1167/053-621-1187 대구광역시 남구  대명로 20길 22-2
'''