from xml.etree.ElementTree import ElementTree

import  httplib2



import httplib2


headers = {"Content-Type": "application/json",
           "Accept": "application/json"}
http = httplib2.Http()

resp, content = http.request("https://bitworking.org/", "GET", headers=headers)

print(resp)
print(content)
