
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random


def get_lhb_data(date_str):
    """
    抓取指定日期的同花顺龙虎榜数据
    :param date_str: 日期字符串，格式为 'YYYY-MM-DD'
    :return: DataFrame 格式的龙虎榜数据
    """
    url = f"http://data.10jqka.com.cn/market/longhu/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "http://data.10jqka.com.cn/market/longhu/"
    }
    
    # 添加参数模拟请求特定日期的数据（实际网站可能需要登录或有反爬机制）
    params = {
        "date": date_str,
        "ajax": "1"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            # print(response.text)
            soup = BeautifulSoup(response.text, 'html.parser')

            # print(soup)
            divs = soup.find_all('div', attrs={'class': 'twrap'})
            # print(divs)
            
            # 解析表格数据

            table = soup.find_all('table', attrs={'class': 'm-table'})


            table = divs[0]
            print(table)
            if not table:
                print("未找到龙虎榜表格")
                return None
                
            rows = table.find_all('tr')[1:]  # 跳过表头
            
            data_list = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 7:  # 确保列数足够
                    stock_code = cols[1].get_text(strip=True)
                    stock_name = cols[2].get_text(strip=True)
                    close_price = cols[3].get_text(strip=True)
                    change_percent = cols[4].get_text(strip=True)
                    turnover = cols[5].get_text(strip=True)
                    net_amount = cols[6].get_text(strip=True)
                    # buy_amount = cols[7].get_text(strip=True)
                    # sell_amount = cols[8].get_text(strip=True)
                    
                    data_list.append({
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '收盘价': close_price,
                        '涨跌幅(%)': change_percent,
                        '成交额(万)': turnover,
                        '净买入额(万)': net_amount
                        # '买入额(万)': buy_amount,
                        # '卖出额(万)': sell_amount
                    })
            
            df = pd.DataFrame(data_list)
            return df
        else:
            print(f"请求失败，状态码：{response.status_code}")
            return None
    except Exception as e:
        print(f"发生异常：{e}")
        return None


def save_to_csv(df, filename="ths_lhb_data.csv"):
    """保存数据到CSV文件"""
    if df is not None and not df.empty:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已保存至 {filename}")
    else:
        print("无数据可保存")


def main():
    # 示例：抓取最近一天的数据
    today = time.strftime('%Y-%m-%d', time.localtime())
    print(f"正在抓取 {today} 的同花顺龙虎榜数据...")
    
    lhb_df = get_lhb_data(today)
    if lhb_df is not None:
        print(lhb_df.head())
        save_to_csv(lhb_df)
    else:
        print("未能获取有效数据")

if __name__ == "__main__":
    main()
