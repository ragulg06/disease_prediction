from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('disease_prediction_model.pkl')

# List of all possible symptoms (same order as training)
symptoms = [
    'itching', 'skin_rash', 'chills', 'joint_pain', 'vomiting', 'fatigue', 'cough',
    'high_fever', 'headache', 'yellowish_skin', 'nausea', 'loss_of_appetite',
    'constipation', 'abdominal_pain', 'diarrhoea', 'yellowing_of_eyes', 'malaise',
    'redness_of_eyes', 'chest_pain', 'neck_pain', 'dizziness', 'slurred_speech',
    'muscle_weakness', 'swelling_joints', 'loss_of_balance', 'bladder_discomfort',
    'irritability', 'muscle_pain', 'increased_appetite', 'receiving_unsterile_injections',
    'coma', 'inflammatory_nails', 'blister'
]

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [0] * len(symptoms)
    for i, symptom in enumerate(symptoms):
        if request.form.get(symptom):
            input_data[i] = 1

    input_df = pd.DataFrame([input_data], columns=symptoms)

    prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction_text=f"Predicted Disease: {prediction}", symptoms=symptoms)

if __name__ == '__main__':
    app.run(debug=True)
