# -*- coding: utf-8 -*-
import requests
cookies="Hm_lvt_740e2ff15c1b3cd6b1101df9015e2ddf=1652844442,1654478388; wxy_ssid=86FE0C6F7C3E6C2AB6BD4E8F1D7A852F; Hm_lpvt_740e2ff15c1b3cd6b1101df9015e2ddf=1654572702; JSESSIONID=25324A7B342468B728C1D364594ECC85"


cookies2 = dict(map(lambda x:x.split('='),cookies.split(";")))
print(cookies2)

s = requests.session()
s.keep_alive = True
for k,v in cookies2.items():
        s.cookies[k]=v

# s.cookies.set_cookie("Hm_lvt_740e2ff15c1b3cd6b1101df9015e2ddf=1652844442,1654478388; wxy_ssid=86FE0C6F7C3E6C2AB6BD4E8F1D7A852F; JSESSIONID=F789E3639421EC940B516750BC3AF7C4; Hm_lpvt_740e2ff15c1b3cd6b1101df9015e2ddf=1654571449")
data = {
"q": "铁路货车",
"fq":"",
"ext":"",
"mp":"",
"sort": 1,
"type": 2,
"page": 1,
"trk": "index"
    }
s.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36'
    }

r = s.post('https://www.wanxiangyun.net/service/search/list',data=data)

cookie = r.cookies
print(cookie)
# print(r.content.decode())
# print(r.status_code )



print(r.content.decode())

