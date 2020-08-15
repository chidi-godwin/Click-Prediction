from flask import Flask, request, jsonify
import pandas as pd 
import numpy as np
import pickle
import dill
import traceback


app = Flask(__name__)

@app.route('/')
def index():
    return "hello world"

@app.route('/predict', methods=['GET', "POST"])
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)

            columns = ['age', 'location_region', 'os_name', 'customer_class', 'customer_value', 'spend_total']
            data = preprocess.fit_transform(pd.DataFrame(json_, columns=columns))

            prediction = list(model.predict(data))

            return jsonify({'prediction': prediction})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print("Train the model first")
        return "No model here to use"


if __name__ == "__main__":
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('preprocessor_pipe.pkl' 'rb') as f:
        preprocessor = dill.load(f)
    
    with open('bins.pkl', 'rb') as f:
        intervals, bin_labels = pickle.load(f)

    app.run(port=5000, debug=False)

