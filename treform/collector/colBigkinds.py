import urllib
import urllib.request
import urllib.parse
import bs4
import json
import re
import sys
import time, random

from treform.collector.colBasic import *

class ColBigkinds(ColBasic):
    def getInfo(params):
        return '빅카인즈 수집기', 0

    def getName(self):
        return "colBigKinds"

    def getSearchURL(self):
        return 'https://www.bigkinds.or.kr/news/detailSearch.do'

    def makeParameter(query, page, startDate, endDate):
        return {
            'pageInfo': 'newsResult',
            'login_chk': 'null',
            'LOGIN_SN': 'null',
            'LOGIN_NAME': 'null',
            'indexName': 'news',
            'keyword': query,
            'byLine': '',
            'searchScope': '1',
            'searchFtr': '1',
            'startDate': startDate,
            'endDate': endDate,
            'sortMethod': 'date',
            'contentLength': '100',
            'providerCode': '',
            'categoryCode': '',
            'incidentCode': '',
            'dateCode': '',
            'highlighting': 'true',
            'sessionUSID': '',
            'sessionUUID': 'test',
            'listMode': '',
            'categoryTab': '',
            'newsId': '',
            'filterProviderCode': '',
            'filterCategoryCode': '',
            'filterIncidentCode': '',
            'filterDateCode': '',
            'filterAnalysisCode': '',
            'startNo': page,
            'resultNumber': '100',
            'topmenuoff': '',
            'resultState': 'detailSearch',
            'keywordJson': '{"searchDetailTxt1":"' + query + '","agreeDetailTxt1":"","needDetailTxt1":"","exceptDetailTxt1":"","o_id":"option1","startDate":"' + startDate + '","endDate":"' + endDate + '","providerNm":"","categoryNm":"","incidentCategoryNm":"","providerCode":"","categoryCode":"","incidentCategoryCode":"","searchFtr":"1","searchScope":"1","searchKeyword":"' + query + '"}',
            'keywordFilterJson': '',
            'interval': '',
            'quotationKeyword1': '',
            'quotationKeyword2': '',
            'quotationKeyword3': '',
            'searchFromUseYN': 'N',
            'mainTodayPersonYn': '',
            'period': ''
        }

