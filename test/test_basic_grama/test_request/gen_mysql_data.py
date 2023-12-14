import json

import requests

def inser_data():
    for i in range(1, 101):
        response = requests.request('GET', 'http://localhost:8080/add')
        print(response.text+" "+str(i))

def get_all_data():
    response = requests.request('GET', 'http://localhost:8080/all') 
    print(response.text )



def get_data_by_parion(pation):
    response = requests.request('GET', 'http://localhost:8080/allByPartion?pation='+pation)
    print(response.text )

if __name__ == '__main__':
    # get_data_by_parion("p2")
    inser_data()