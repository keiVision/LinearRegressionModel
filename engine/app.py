from model import *
from data_processor import *
from config import *
from main import * 

from sklearn.preprocessing import OneHotEncoder
from flask import Flask, request, jsonify, redirect
from sklearn.model_selection import train_test_split
from flask_cors import CORS

import pandas as pd

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Train model once at app initialization
    engine = Engine()
    engine.train_model()
    engine.check_train_status()

    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type, Access-Control-Allow-Headers'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response

    def add_header(response):
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        input_data = pd.DataFrame(data=[data], columns=['Object_area', 'Process_name'])

        predicted_days, predicted_volume = engine.predict(input_data)

        result = {predicted_days, predicted_volume}

        return jsonify(result)

    @app.route('/', methods=['GET'])
    def https_redirect():
        return redirect("https://194-58-98-29.cloudvps.regruhosting.ru:5000/predict")

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = True)
