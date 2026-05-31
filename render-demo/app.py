from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# --- গ্লোবালি model.pkl লোড করা হচ্ছে ---
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("✅ Model loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: Could not find model.pkl. Details: {e}")
    model = None

# --- আপনার জুপিটার নোটবুকের সেই আসল ২২টি কলামের লিস্ট ---
# (নোটবুকের X_train.columns-এর সাথে মিলিয়ে এই ২২টি নাম নিশ্চিত করুন)
MODEL_FEATURES = [
    "age", 
    "trestbps", 
    "chol", 
    "thalch", 
    "oldpeak", 
    "ca",
    "sex_female", "sex_male",
    "cp_asymptomatic", "cp_atypical angina", "cp_non-anginal", "cp_typical angina",
    "fbs_false", "fbs_true",
    "restecg_left ventricular hypertrophy", "restecg_normal", "restecg_st-t wave abnormality",
    "exang_false", "exang_true",
    "slope_flat", "slope_upsloping",
    "thal_normal"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html", 
            prediction_text="Error: Model file is missing on the server."
        )

    try:
        # ১. ফর্ম থেকে ডেটা কালেক্ট করা
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

        # ২. ডেটাফ্রেমে রূপান্তর ও ওয়ান-হট এনকোডিং
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)

        # ৩. ম্যানুয়াল কলাম ম্যাচিং (২২টি কলামের কাঠামো তৈরি করা)
        # কারেন্ট ফর্মে যে কলামগুলো তৈরি হয়নি, সেগুলোকে ০ ধরা হবে
        for col in MODEL_FEATURES:
            if col not in df.columns:
                df[col] = 0
        
        # অতিরিক্ত কলাম (যেমন dataset) বাদ দিয়ে হুবহু ২২টি কলামের সঠিক সিকুয়েন্স করা
        df = df[MODEL_FEATURES]

        # ৪. প্রেডিকশন এবং প্রবাবিলিটি বের করা
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        if prediction == 1:
            result = f"Heart Disease Detected ({probability*100:.2f}% risk)"
        else:
            result = f"No Heart Disease ({(1-probability)*100:.2f}% confidence)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Runtime Error: {str(e)}")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)