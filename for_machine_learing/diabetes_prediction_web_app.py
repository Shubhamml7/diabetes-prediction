# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:57:01 2025

@author: ShivamShubham
"""
import numpy as np
import pickle 
import streamlit as st

# Load model
loaded_model = pickle.load(open(
    'C:/Users/ShivamShubham/Desktop/for machine learning/trained_model.sav', 'rb'
))

# Prediction function
def diabetes_prediction(input_data):

    # Convert input to numpy array (all floats)
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Predict
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'Person is NOT diabetic'
    else:
        return 'Person IS diabetic'


def main():
    st.title('Diabetes Prediction Web App')

    # User input fields
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the Person")

    diagnosis = ''

    if st.button("DIABETES TEST RESULT"):
        try:
            user_input = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]

            diagnosis = diabetes_prediction(user_input)

        except ValueError:
            diagnosis = "⚠️ Please enter **numeric values only**."

    st.success(diagnosis)


if __name__ == "__main__":
    main()
    
         
     
     
     