import joblib
import pandas as pd

# Load the saved model
model = joblib.load("models/diabetes_model.pkl")

# Sample patient data
patient = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 25.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}])

# Make prediction
prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0][1]

if prediction == 1:
    print("Model prediction: Higher diabetes risk")
else:
    print("Model prediction: Lower diabetes risk")

print("Estimated model probability:", round(probability * 100, 2), "%")