from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and feature columns
model = joblib.load("savedmodel\\model.pkl")           # or "crop_connect.pkl"
columns = joblib.load("savedmodel\\X_columns.pkl")     

@app.route('/')
def home():
    return "Smart Irrigation API is running! 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    new_df = pd.DataFrame([data])

    new_df = pd.get_dummies(new_df, columns=["Crop", "Growth Stage", "Soil Type", "Climate Zone"], drop_first=True)

    new_df = new_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(new_df)[0]

    if prediction < 30:
        status = "Bad"
    elif 30 <= prediction <= 50:
        status = "Average"
    else:
        status = "Good"

    return jsonify({
        "moisture": round(prediction, 2),
        "status": status
    })

if __name__ == '__main__':
    app.run(debug=True)
