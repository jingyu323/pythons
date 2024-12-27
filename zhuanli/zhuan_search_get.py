# -*- coding: utf-8 -*-
import json
import xlwings as xw
import sqlite3
from bs4 import BeautifulSoup

import requests
cookies="Hm_lvt_740e2ff15c1b3cd6b1101df9015e2ddf=1652844442,1654478388; wxy_ssid=86FE0C6F7C3E6C2AB6BD4E8F1D7A852F; Hm_lpvt_740e2ff15c1b3cd6b1101df9015e2ddf=1654572702; JSESSIONID=25324A7B342468B728C1D364594ECC85"
cookies2 = dict(map(lambda x:x.split('='),cookies.split(";")))
print(cookies2)

s = requests.session()
s.keep_alive = True
for k,v in cookies2.items():
        s.cookies[k]=v

# s.cookies.set_cookie("Hm_lvt_740e2ff15c1b3cd6b1101df9015e2ddf=1652844442,1654478388; wxy_ssid=86FE0C6F7C3E6C2AB6BD4E8F1D7A852F; JSESSIONID=F789E3639421EC940B516750BC3AF7C4; Hm_lpvt_740e2ff15c1b3cd6b1101df9015e2ddf=1654571449")
s.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Cookie': 'Hm_lvt_740e2ff15c1b3cd6b1101df9015e2ddf=1652844442,1654478388; wxy_ssid=86FE0C6F7C3E6C2AB6BD4E8F1D7A852F; Hm_lpvt_740e2ff15c1b3cd6b1101df9015e2ddf=1654572702; JSESSIONID=25324A7B342468B728C1D364594ECC85',
    }

current_page=1
url='https://www.wanxiangyun.net/service/search/list?q=IP网络物理&sort=1&type=2&trk=index&page=' + str(current_page)
r = s.post(url)

cookie = r.cookies

print(r.content.decode())
# print(r.status_code )

page_size=20
data2 = json.loads(r.text)
total_count=int(data2["total_count"])
print(data2["total_count"])
print(data2["patents"])
total_page= (total_count-1)//page_size +1


workbook = xw.Book("d:\\专利.xlsx") #连接excel文件

#
# workbook.save('专利.xlsx')
# workbook.close()
titles = ["专利名称", "摘要", "申请号", "公开号", "最早公开日", "申请日期", "申请人","状态","代理机构"]

sheet1 = workbook.sheets["IP网络物理"]
print(sheet1.name)
sheet1.range("A1").value=titles
index=2
while current_page <= total_page:
    r = s.post(url)
    print(r.text)
    data2 = json.loads(r.text)
    for iten   in data2["patents"]:
        row=[]
        #专利名称
        title = iten['title']['original']
        row.append(title)
        # 摘要
        if iten['abstract']['original'] == "en":
            abstract = iten['abstract']['zh-cn']
            if abstract == "":
                abstract = iten['abstract']['original']
            row.append(abstract)
        else:
            abstract = iten['abstract']['original']
            row.append(abstract)
        #申请号 application_number
        application_number = iten['application_number']
        row.append(application_number)
        #公开号
        publication_history = iten['publication_history'][0]
        row.append(publication_history)

        # earliest_publication_date 最早公开日
        earliest_publication_date = iten['earliest_publication_date']
        row.append(earliest_publication_date)
        #申请日期
        application_date = iten['application_date']
        row.append(application_date)
        # 公司名称
        if  len(iten['applicants']) > 0:
            com_name=iten['applicants'][0]['name']['original']
            print(com_name)
            row.append(com_name)
        else:
            row.append("")

        # 状态
        status =iten['legal_status']
        pls =int(iten['pls'])
        prl =iten['prl']
        free_to_operate =iten['free_to_operate']
        status_str=""
        if status !="":
            if status != "3" and status != "1" and  pls > 0  and prl > 0:
                status_str= status_str + "有效:剩余"+str(prl)+"年"
            else:
                status_str = status_str + "失效"

        if free_to_operate == 1:
            status_str = status_str +  " 非公用"
        elif free_to_operate == 2:
            status_str = status_str + " 公用"
        else:
            status_str = status_str + " 未知公用"
        row.append(status_str)
        # 代理机构

        agencies=iten.get("agencies","")
        if agencies != "":
            row.append(agencies[0])
        else:
            row.append("")
        rownum  = "A"+str(index)

        sheet1.range(rownum).value = row
        index = index +1
    current_page = current_page +1
