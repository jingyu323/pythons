from urllib.robotparser import RobotFileParser
rp = RobotFileParser()
rp.set_url('http://www.jianshu.com/robots.txt')
rp.read()

print(rp)
print(rp.can_fetch('*','http://www.jianshu.com/p/b67554025d7d'))
print(rp.can_fetch('*',"http://www.jianshu.com/search?q=python&page=1&type=collections"))


import requests

data = {
'name': 'germey',
'age': 22}

r = requests.get('https://www.baidu.com/',params=data)
print(type(r))
print(r.status_code)
print (type(r.text))
print(r.text)
print(r.cookies)

print("============r1==========")

r1 = requests.post('http://httpbin.org/post')
print(type(r1))
print(r1.status_code)
print (type(r1.text))
print(r1.text)
print(r1.cookies)