from flask import Flask, render_template, request
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import numpy as np
import joblib


app = Flask(__name__)

# model loaded
model = tf.keras.models.load_model('health_model.keras')
scaler = joblib.load('scaler.pkl')

@app.route('/')

def index():

    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])

def predict():
    if request.method == "POST":

        columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        form_data = request.form.to_dict()
        data_df = pd.DataFrame([form_data])

        scaled_input = scaler.transform(data_df)


        data_df = data_df[columns]


        prediction = model.predict(scaled_input)
        predicted_cost = float(prediction[0][0])

        return render_template("index.html", prediction_text = "The cost is {}".format(predicted_cost))


if __name__ == "__main__":
    app.run(debug=True)
