import streamlit as st
import pandas as pd
import joblib
import numpy as np


try:
    model = joblib.load('random_forest_salary_model.pkl')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'random_forest_salary_model.pkl' not found. "
             "Please ensure it's in the same directory as this app.py file.")
    st.stop() 


try:
    original_df = pd.read_csv('Salary Data.csv')

    
    original_df.dropna(subset=['Salary'], inplace=True)
    for col in ['Age', 'Years of Experience']:
        if original_df[col].isnull().any():
            median_val = original_df[col].median()
            original_df[col].fillna(median_val, inplace=True)
    for col in ['Gender', 'Education Level', 'Job Title']:
        if original_df[col].isnull().any():
            original_df[col].fillna('Unknown', inplace=True)

    
    dummy_encoded_df = pd.get_dummies(original_df.drop('Salary', axis=1),
                                      columns=['Gender', 'Education Level', 'Job Title'],
                                      drop_first=True)
    model_features = dummy_encoded_df.columns 

except FileNotFoundError:
    st.error("Error: Original data file 'Salary Data.csv' not found. "
             "This file is needed to ensure consistent preprocessing. "
             "Please ensure it's in the same directory as this app.py file.")
    st.stop() # Stop the app if the data can't be loaded for feature alignment
except Exception as e:
    st.error(f"An error occurred during feature preparation: {e}")
    st.stop()


# --- 3. Streamlit App Layout and User Input ---
st.title("💰 Employee Salary Predictor")
st.write("Enter employee details below to get an estimated salary prediction.")

# Input fields for user
age = st.slider("Age", 18, 65, 30, help="Employee's age in years.")
gender = st.selectbox("Gender", ["Male", "Female"], help="Employee's gender.")
education_level = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"], help="Highest education qualification.")
job_title = st.text_input("Job Title", "Software Engineer", help="Enter the exact Job Title (e.g., 'Software Engineer', 'Data Analyst'). Case-sensitive.")
years_experience = st.slider("Years of Experience", 0, 40, 5, help="Number of years of professional experience.")

# --- 4. Prediction Logic ---
if st.button("Predict Salary 📈"):
    # Create a DataFrame from user input, matching the structure of the original data BEFORE one-hot encoding
    input_data = pd.DataFrame([[age, gender, education_level, job_title, years_experience]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Apply the same one-hot encoding to the user's input
    # This will create new columns based on the input's categorical values.
    input_encoded = pd.get_dummies(input_data,
                                   columns=['Gender', 'Education Level', 'Job Title'],
                                   drop_first=True)

    # Align columns of user input with the columns the model was trained on.
    # Create an empty DataFrame with all expected model features, initialized to 0.
    final_input = pd.DataFrame(0, index=[0], columns=model_features)

    # Fill in the values from the user's encoded input into the final_input DataFrame.
    # Any column present in input_encoded but not in model_features (e.g., a completely new job title)
    # will be ignored, effectively treating it as all 0s for its one-hot encoding.
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col]
        # If 'col' is not in 'final_input.columns', it means this specific category
        # was not present during training or was dropped (e.g., 'Female' if drop_first=True for Gender).
        # Since final_input is initialized with zeros, these missing columns automatically default to zero,
        # which is the correct behavior for unseen categorical values in one-hot encoding.


    # Ensure the order of columns in final_input exactly matches model_features
    # This is critical for the model to make correct predictions.
    final_input = final_input[model_features]

    # Make prediction using the loaded model
    try:
        predicted_salary = model.predict(final_input)[0]
        st.success(f"### Estimated Salary: **₹{predicted_salary:,.2f}**")
        st.info("💡 **Note:** The accuracy of prediction for 'Job Title' depends on whether it was present in the training data.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your inputs, especially the 'Job Title', to ensure consistency with the training data.")