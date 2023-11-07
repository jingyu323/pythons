import requests


for i in range(1, 101):
    response = requests.request('GET', 'http://localhost:8080/add')