#-*- coding: euc-kr -*-

import requests
from bs4 import BeautifulSoup
import re, json, os, random
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import pandas as pd
from io import StringIO


def task(page, max_pages, url_path, file_w, reg=None, url_list=[]):
    """
    문서내의 또다른 Url 을 찾아서 반복 수행 Task
    """
    if page == max_pages:
        get_single_article(url_path, file_w, reg_exp=str(reg))
    else:
        get_single_article(url_path, file_w, reg_exp=str(reg))
        source_code = requests.get(url_path)
        plain_text = source_code.text
        soup = BeautifulSoup(plain_text, 'lxml')
        page += 1
        for link in soup.find_all('a'):
            href = link.get('href')
            if (href != None and re.search("https://ko", href) and href not in url_list):
                url_list.append(href)
                task(page, max_pages, href, file_w, reg=str(reg), url_list=[])


def get_single_article(item_url, file_w, reg_exp=None):
    """
    요청된 단일 Request 처리 Function
    """

    if (reg_exp == 'table') :
        table_to_csv(item_url)
    elif (reg_exp):
        print("href : {0}".format(item_url))
        source_code = requests.get(item_url)
        plain_text = source_code.text
        soup = BeautifulSoup(plain_text, 'lxml')
        reg = re.compile(reg_exp)
        for contents in reg.findall(plain_text):
            file_w.write(contents)
        file_w.write('\\n')

def spider(max_pages, url_path, path="/home/dev/wiki/", file_name='test.txt', reg_exp=None):
    """
    Main Function
    """
    if not os.path.exists(path):
        os.makedirs(path)
    with open(''.join([path, file_name]), "w") as file_w:
        print("# Crawler Job Start!!")
        task(1, max_pages, url_path, file_w, reg=reg_exp, url_list=[])
        print("# Crawler Job Done!!")


def save_as_csv(data):
    """
    csv 형태로 Local 에 저장
    """
    rand_name = random.randrange(1, 10000)
    save_data = StringIO(data)
    df = pd.read_csv(save_data, sep=",")
    df.to_csv("/home/dev/csv/" + str(rand_name) + ".csv", sep=',', encoding='utf-8')
    print("file saved as : {0}".format(str(rand_name)))


def table_to_csv(url):
    """
    table 구조를 Parse 해서 csv 로 변환
    """
    try:
        return_line = []
        return_td = ""
        source_code = requests.get(url)
        plain_text = source_code.text
        soup = BeautifulSoup(plain_text, 'lxml')
        for table in soup.find_all('table'):
            for tr in table.find_all('tr'):
                for td in table.find_all('td'):
                    return_td = return_td + td.text + ','
                return_td = return_td.rstrip(',') + '\\n'
            # save each table
            save_as_csv(return_td)
            return_td = ""
    except Exception as e:
        return True



