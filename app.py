import streamlit as st
import pickle
import pandas as pd

# load model
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("House Price Prediction App")

inputs = {}

for col in features:
    inputs[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    prediction = model.predict(df)[0]
    st.success(f"Predicted Price: {prediction}")