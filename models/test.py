import requests

url = "http://0.0.0.0:8989/"


car_specifics =  {
'gearbox' : ['manual'],
'fuelType': ['benzin'],
'notRepairedDamage' : ['no']
}

r = requests.post(url, json = car_specifics)
print(r.text)
