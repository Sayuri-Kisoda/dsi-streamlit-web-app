# Import Libraries

import streamlit as st 
import pandas as pd
import joblib


# Load Our Pipeline Object

model = joblib.load("model.joblib")


# Add Title and Instructions

st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")


# Age Input Form
age = st.number_input(
    label = "01. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value = 35)

# Gender Input Form
gender = st.radio(
    label = "02. Enter the customer's gender",
    options = ['M', 'F'],
    )


# Credit Score Input Form
credit_score = st.number_input(
    label = "03. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000,
    value = 500)


# Submit Inputs to Model

if st.button("Submit For Prediction"):
    
    # store our data in a dataframe for prediction
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})

    # apply model pipeline to the input data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    
    # output prediction
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}")
                          









