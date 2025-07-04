from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("car_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.form.to_dict()
        for key in input_data:
            try:
                input_data[key] = float(input_data[key])
            except ValueError:
                pass
        df = pd.DataFrame([input_data])
        prediction_log = model.predict(df)[0]
        prediction = np.expm1(prediction_log)
        return render_template("index.html", predicted_price=f"${prediction:,.2f}")
    except Exception as e:
        return render_template("index.html", predicted_price=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
