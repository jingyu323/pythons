from urllib.robotparser import RobotFileParser


import requests


def get_test():
    rp = RobotFileParser()
    rp.set_url('http://www.jianshu.com/robots.txt')
    rp.read()

    print(rp)
    print(rp.can_fetch('*', 'http://www.jianshu.com/p/b67554025d7d'))
    print(rp.can_fetch('*', "http://www.jianshu.com/search?q=python&page=1&type=collections"))

    data = {
        'name': 'germey',
        'age': 22}

    r = requests.get('http://httpbin.org/get', params=data)
    print(type(r))
    print(r.status_code)
    print(type(r.text))
    print(r.text)
    print(r.cookies)
    print(type(r.json()))
    print(r.json())

    print("============r1==========")

    r1 = requests.post('http://httpbin.org/post')
    print(type(r1))
    print(r1.status_code)
    print(type(r1.json()))
    print(r1.json())
    print(r1.cookies)

    r = requests.get("https://github.com/favicon.ico")
    with open('favicon.ico', 'wb') as f:
        f.write(r.content)


def zhihu():
    print("============zhihu==========")

    headers = {
        "Cookie":"_zap=27f77497-51d4-4372-ab3b-058bf25013c3; d_c0=AHCYXRXk6BWPTh7D-l4hZrFYznu9af2Lsg0=|1669186300; _xsrf=mYiHc1YhgUTYGNE0XnJoxX2gffRZtjb9; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1687136965,1687655579,1687658627,1687742704; captcha_session_v2=2|1:0|10:1687742712|18:captcha_session_v2|88:cExKRHlJSVVLWkNNbGxDbnlORjVoUG4vY2hiTlBlT1BFeFZqdHY0c0ZIVnhBaDA2SGpUdUhWUkxzS1NzVmhSTg==|499c5b229dfaedc92d68f6df9f14ef5887b79e3496ff70d25bec9dbcc70c7bbf; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1687758862",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    r = requests.get("https://www.zhihu.com/explore", headers=headers)
    print(r.text)


def cookie_test():
    r = requests.get("https://www.baidu.com")
    print(r.cookies)
    for  key, value in  r.cookies.items():
        print(key + '=' + value)


def ssl_test():
    response = requests.get('https://www.12306.cn')
    print(response)
    print(response.status_code)


def test_auth():
    r = requests.get("http://localhost:1081/#/login",auth=('admin','test'))
    print(r.text)
    print(r.status_code)
# Prepared Request
from requests import Request, Session

def Preparedreq_test():
    url="http://httpbin.org/post"

    data = { 'name':'germey '}
    headers1 = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    s = Session()
    req = Request("POST",  url, data = data, headers=headers1)

    prepped = s.prepare_request(req)
    r = s.send(prepped)
    print(r.text)





if __name__=='__main__':
    Preparedreq_test()