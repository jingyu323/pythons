

import json
import pandas as pd
import requests


def get_data():
    url = 'https://applhb.longhuvip.com/w1/api/index.php'
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
    response = requests.post(url, params=params, headers=headers)
    # 将编码设置为当前编码
    response.encoding = response.apparent_encoding
    # 解析JSON数据
    data = json.loads(response.text)
    # 获取买入营业部、卖出营业部和风口概念等数据
    BIcon = data.get('BIcon')
    SIcon = data.get('SIcon')
    fkgn = data.get('fkgn')
    lb = data.get('lb')
    all_data = pd.DataFrame()
    # 遍历股票列表，提取数据
    for item in data.get('list'):
        ID = item.get('ID')
        item_data = [
            ID,
            item.get('Name'),
            item.get('IncreaseAmount'),
            item.get('BuyIn'),
            item.get('JoinNum'),
            ','.join(BIcon.get(ID, [])),
            ','.join(SIcon.get(ID, [])),
            ','.join(fkgn.get(ID, {}).values()),
            lb.get(ID),
        ]

        # 将数据转换成DataFrame类型
        data = pd.DataFrame(item_data).T
        dict = {0:'股票代码', 1:'股票名称', 2:'涨幅', 3:'净买入', 4:'关联数', 5:'买入营业部', 6:'卖出营业部', 7:'风口概念', 8:'连板数'}
        data.rename(columns=dict, inplace=True)
        all_data = all_data.append(data, ignore_index=True)
    # 返回DataFrame类型数据



if __name__ == '__main__':
    df = get_data()
    df.to_csv('longhubang.csv',encoding='gbk')
    print(df)