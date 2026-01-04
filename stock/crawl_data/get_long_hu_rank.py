

import json

import pandas
import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_data():
    url = 'http://data.10jqka.com.cn/market/longhu/'
    headers = {
        'user-agent':'Mozilla/5.0(Linux; Android 7.1.2; SM-G955N Build/NRD90M.G955NKSU1AQDC; wv)'
    }
    # POST请求参数
    params = {
        'st': '500',
        'Index': '0',
        'c': 'LongHuBang',
        'PhoneOSNew': 1,
        'a': 'GetStockList',
        'DeviceID': '0f6ac4ae-370d-3091-a618-1d9dbb2ecce0',
        'apiv': 'w31',
        'Type': 2,
        'UserID': 0,
        'Token': 0,
        'Time': 0,
    }
    # 发送POST请求
    response = requests.get(url, params=params, headers=headers)
    # 将编码设置为当前编码
    response.encoding = response.apparent_encoding
    # 解析JSON数据

    soup = BeautifulSoup(response.text, 'lxml')

    page_table = soup.find('div', attrs={'class': 'page-table'})
    div_wrap = page_table.find('div', attrs={'class': 'twrap'})
    table = div_wrap.find('table', attrs={'class': 'm-table'})
    all_data = pd.DataFrame()

    if table:
        tbody = table.find('tbody')  # 跳过表头
        rows = tbody.find_all('tr')  # 跳过表头
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 0:
                stock_data = [
                     cols[1].text.strip(),
                     cols[2].text.strip(),
                     cols[4].text.strip(),
                     cols[5].text.strip(),
                    0,
                    "B",
                    "s",
                    "t",
                    cols[0].text.strip(),
                    cols[3].text.strip(),

                ]
                data = pd.DataFrame(stock_data).T
                dict = {0: '股票代码', 1: '股票名称', 2: '涨幅', 3: '净买入', 4: '关联数', 5: '买入营业部', 6: '卖出营业部',
                        7: '风口概念', 8: '连板数', 9: '现价'}
                # dict = {0: '股票代码', 1: '股票名称'}
                data.rename(columns=dict, inplace=True)
                all_data = pandas.concat([all_data,data], axis=0)

    return all_data
    # 返回DataFrame类型数据



if __name__ == '__main__':
    df = get_data()
    df.to_csv('longhubang.csv',encoding='gbk')


    print(df)