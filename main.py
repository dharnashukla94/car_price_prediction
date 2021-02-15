
import pickle
from flask import Flask, request, jsonify
from models.ml_model import runner


app = Flask("Car_Prediction")

@app.route('/', methods = ['POST'])
def predict():
    car_specifics= request.get_json()
    price = runner(car_specifics)
    response = {' Price' : list(price)}
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 8989)
