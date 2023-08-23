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

    engine = Engine()

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
        input_data = pd.DataFrame([data], columns = ['Object_area', 'Process_name', 'Directive_perfomance', 'Hour_cost'])
        hum_days, process_volume = engine.predict(input_data)


        process_name = input_data['Process_name'].iloc[0]
        object_area = input_data['Object_area'].iloc[0]
        hum_count = int(hum_days) // int(input_data['Directive_perfomance'].iloc[0])
        final_price = int(float((hum_days * 10) * float(input_data['Hour_cost'].iloc[0])))
        hour_cost = input_data['Hour_cost'].iloc[0]
        directive_perfomance = input_data['Directive_perfomance'].iloc[0]

        result = {'Process_name': str(process_name), 
                  'Object_area': int(object_area), 
                  'Hum_hours_cost': (hour_cost),
                  'Hum_count': int(hum_count),
                  'Hum_days': int(hum_days),
                  'Final_price': int(final_price),
                  'Process_volume': int(process_volume),
                  'Directive_perfomance': int(directive_perfomance)}

        return jsonify(result)

    @app.route('/', methods=['GET'])
    def https_redirect():
        return redirect("https://194-58-98-29.cloudvps.regruhosting.ru:5000/predict")

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, debug = True)
