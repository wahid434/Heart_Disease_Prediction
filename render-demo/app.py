from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# load trained model
model = pickle.load(open("heart_disease_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "age": float(request.form["age"]),
        "trestbps": float(request.form["trestbps"]),
        "chol": float(request.form["chol"]),
        "thalch": float(request.form["thalch"]),
        "oldpeak": float(request.form["oldpeak"]),
        "ca": float(request.form["ca"]),

        "sex": request.form["sex"],
        "cp": request.form["cp"],
        "fbs": request.form["fbs"],
        "restecg": request.form["restecg"],
        "exang": request.form["exang"],
        "slope": request.form["slope"],
        "thal": request.form["thal"],
        "dataset": request.form["dataset"],
    }

    df = pd.DataFrame([data])

    # convert to one-hot exactly like training
    df = pd.get_dummies(df)

    # match training columns
    model_columns = pickle.load(open("model_columns.pkl", "rb"))

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        result = f"Heart Disease Detected ({probability*100:.2f}% risk)"
    else:
        result = f"No Heart Disease ({(1-probability)*100:.2f}% confidence)"

    return render_template(
        "index.html",
        prediction_text=result
    )


if __name__ == "__main__":
    app.run(debug=True)
    