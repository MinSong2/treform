import os
import sys
import requests
import json
import csv

client_id = "___________to be filled___________"
client_secret = "___________to be filled___________"

csv_file_path = 'naver_news1.csv'
csvfile = open(csv_file_path, 'a+', newline='', encoding='utf-8')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(["title", "origiallink", "link", "description", "pubDate"])

headers = {'X-Naver-Client-Id': client_id,
                   'X-Naver-Client-Secret':client_secret}

max_page=2
#http protocol
encText = "대통령"
for x in range(1,max_page+1):
    try:
        url = "https://openapi.naver.com/v1/search/news.json?query=" + encText + "&start=" + str(x) + "&display=" + str(100) # json 결과
        #url = "https://openapi.naver.com/v1/search/blog.json?query=" + encText # xml 결과
        print(url)

        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        if response.status_code == 200:
            data = response.json()
            #data = json.loads(result)
            print(data)
            for object in data['items']:
                writer.writerow([object['title'], object['originallink'], object['link'],
                    object['description'], object['pubDate']])
        else:
            print("Error Code:" + str(response.status_code))
    except IndexError:
        print('out of index error')