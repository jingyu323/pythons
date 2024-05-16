# -*- coding:UTF-8 -*-
import requests

target = 'https://www.wanxiangyun.net/search/list?q=车载网关产品&trk=index'
req = requests.get(url=target)
print(req.text)