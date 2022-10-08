# -*- coding: utf-8 -*-
import json
import re
import requests
import datetime
import os

from bs4 import BeautifulSoup


def crawl_wiki_data():
    """
    爬取百度百科中《乘风破浪的姐姐》中嘉宾信息，返回html
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
    }
    url = 'https://baike.baidu.com/item/乘风破浪的姐姐'

    try:
        response = requests.get(url, headers=headers)
        # 将一段文档传入BeautifulSoup的构造方法,就能得到一个文档的对象, 可以传入一段字符串
        soup = BeautifulSoup(response.text, 'lxml')

        # 返回所有的<table>所有标签
        tables = soup.find_all('table')
        crawl_table_title = "按姓氏首字母排序"
        for table in tables:
            # 对当前节点前面的标签和字符串进行查找
            table_titles = table.find_previous('div')
            for title in table_titles:
                if (crawl_table_title in title):
                    return table
    except Exception as e:
        print(e)


def parse_wiki_data(table_html):
    '''
    解析得到选手信息，包括包括选手姓名和选手个人百度百科页面链接，存JSON文件,保存到work目录下
    '''
    bs = BeautifulSoup(str(table_html), 'lxml')
    all_trs = bs.find_all('tr')

    stars = []
    for tr in all_trs:
        all_tds = tr.find_all('td')  # tr下面所有的td

        for td in all_tds:
            # star存储选手信息，包括选手姓名和选手个人百度百科页面链接
            star = {}
            if td.find('a'):
                # 找选手名称和选手百度百科连接
                if td.find_next('a'):
                    star["name"] = td.find_next('a').text
                    star['link'] = 'https://baike.baidu.com' + td.find_next('a').get('href')

                elif td.find_next('div'):
                    star["name"] = td.find_next('div').find('a').text
                    star['link'] = 'https://baike.baidu.com' + td.find_next('div').find('a').get('href')
                stars.append(star)

    json_data = json.loads(str(stars).replace("\'", "\""))
    with open('work/' + 'stars.json', 'w', encoding='UTF-8') as f:
        json.dump(json_data, f, ensure_ascii=False)


def crawl_everyone_wiki_urls():
    '''
    爬取每个选手的百度百科图片，并保存
    '''
    with open('work/' + 'stars.json', 'r', encoding='UTF-8') as file:
        json_array = json.loads(file.read())
    headers = {
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
    }
    star_infos = []
    for star in json_array:
        star_info = {}
        name = star['name']
        link = star['link']
        star_info['name'] = name
        # 向选手个人百度百科发送一个http get请求
        response = requests.get(link, headers=headers)
        # 将一段文档传入BeautifulSoup的构造方法,就能得到一个文档的对象
        bs = BeautifulSoup(str(response.text), 'lxml')
        # 获取选手的民族、星座、血型、体重等信息
        base_info_div = bs.find('div', {'class': 'basic-info'})
        dls = base_info_div.find_all('dl')
        for dl in dls:
            dts = dl.find_all('dt')
            for dt in dts:
                if "".join(str(dt.text).split()) == '民族':
                    star_info['nation'] = dt.find_next('dd').text
                if "".join(str(dt.text).split()) == '星座':
                    star_info['constellation'] = dt.find_next('dd').text
                if "".join(str(dt.text).split()) == '血型':
                    star_info['blood_type'] = dt.find_next('dd').text
                if "".join(str(dt.text).split()) == '身高':
                    height_str = str(dt.find_next('dd').text)
                    star_info['height'] = str(height_str[0:height_str.rfind('cm')]).replace("\n", "")
                if "".join(str(dt.text).split()) == '体重':
                    star_info['weight'] = str(dt.find_next('dd').text).replace("\n", "")
                if "".join(str(dt.text).split()) == '出生日期':
                    birth_day_str = str(dt.find_next('dd').text).replace("\n", "")
                    if '年' in birth_day_str:
                        star_info['birth_day'] = birth_day_str[0:birth_day_str.rfind('年')]
        star_infos.append(star_info)

        # 从个人百度百科页面中解析得到一个链接，该链接指向选手图片列表页面
        if bs.select('.summary-pic a'):
            pic_list_url = bs.select('.summary-pic a')[0].get('href')
            pic_list_url = 'https://baike.baidu.com' + pic_list_url

        # 向选手图片列表页面发送http get请求
        pic_list_response = requests.get(pic_list_url, headers=headers)

        # 对选手图片列表页面进行解析，获取所有图片链接
        bs = BeautifulSoup(pic_list_response.text, 'lxml')
        pic_list_html = bs.select('.pic-list img ')
        pic_urls = []
        for pic_html in pic_list_html:
            pic_url = pic_html.get('src')
            pic_urls.append(pic_url)
        # 根据图片链接列表pic_urls, 下载所有图片，保存在以name命名的文件夹中
        down_save_pic(name, pic_urls)
        # 将个人信息存储到json文件中
        json_data = json.loads(str(star_infos).replace("\'", "\"").replace("\\xa0", ""))
        with open('work/' + 'stars_info.json', 'w', encoding='UTF-8') as f:
            json.dump(json_data, f, ensure_ascii=False)

def down_save_pic(name,pic_urls):
    '''
    根据图片链接列表pic_urls, 下载所有图片，保存在以name命名的文件夹中,
    '''
    path = 'work/'+'pics/'+name+'/'
    if not os.path.exists(path):
      os.makedirs(path)

    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            string = str(i + 1) + '.jpg'
            with open(path+string, 'wb') as f:
                f.write(pic.content)
                #print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            #print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue




if __name__ == '__main__':

     #爬取百度百科中《乘风破浪的姐姐》中参赛选手信息，返回html
     html = crawl_wiki_data()

     #解析html,得到选手信息，保存为json文件
     parse_wiki_data(html)

     #从每个选手的百度百科页面上爬取,并保存
     crawl_everyone_wiki_urls()

     print("所有信息爬取完成！")