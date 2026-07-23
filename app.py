"""
Working Prototype — Bengaluru House Price Predictor
AutoML & Hyperparameter Optimization Project

This is a Streamlit web app that loads your trained model and lets users
input house features to get a live price prediction.

HOW TO RUN:
1. First, save your trained model from your Colab notebook (see save_model.py)
2. Download 'best_model.pkl' and 'model_columns.pkl' to the same folder as this file
3. Install streamlit:  pip install streamlit
4. Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(page_title="Bengaluru House Price Predictor", layout="centered")

st.title("🏠 Bengaluru House Price Predictor")
st.write("A working prototype built on top of an AutoML / Hyperparameter-Optimized "
         "Random Forest model.")

# =====================================================================
# LOAD MODEL AND COLUMN REFERENCE
# =====================================================================

def load_model():
    import os
    import pickle

    st.write("Current folder:", os.getcwd())
    st.write("Files:", os.listdir("."))

    with open("best_model.pkl", "rb") as f:
        first10 = f.read(10)
        st.write("best_model first bytes:", first10)
        f.seek(0)
        model = pickle.load(f)

    with open("model_columns.pkl", "rb") as f:
        first10 = f.read(10)
        st.write("model_columns first bytes:", first10)
        f.seek(0)

        try:
            columns = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model_columns.pkl: {e}")
            raise

    return model, columns 
        
try:
    model, model_columns = load_model()
    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")
    st.exception(e)
# =====================================================================
# USER INPUT FORM
# =====================================================================
if model_loaded:
    st.subheader("Enter property details")

    col1, col2 = st.columns(2)

    with col1:
        total_sqft = st.number_input("Total square feet", min_value=300, max_value=10000, value=1200, step=50)
        bhk = st.number_input("BHK (bedrooms)", min_value=1, max_value=10, value=2, step=1)

    with col2:
        bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        balcony = st.number_input("Balconies", min_value=0, max_value=5, value=1, step=1)

    # Extract location options directly from the model's training columns
    location_options = sorted([
        col.replace("location_", "") for col in model_columns if col.startswith("location_")
    ])
    location = st.selectbox("Location", options=location_options)

    area_type_options = sorted([
        col.replace("area_type_", "") for col in model_columns if col.startswith("area_type_")
    ])
    area_type = st.selectbox("Area type", options=area_type_options)

    # =====================================================================
    # PREDICT BUTTON
    # =====================================================================
    if st.button("Predict Price"):
        # Build a single-row DataFrame matching the model's expected columns
        input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

        input_data["total_sqft"] = total_sqft
        input_data["bath"] = bath
        input_data["balcony"] = balcony
        input_data["bhk"] = bhk

        loc_col = f"location_{location}"
        if loc_col in input_data.columns:
            input_data[loc_col] = 1

        area_col = f"area_type_{area_type}"
        if area_col in input_data.columns:
            input_data[area_col] = 1

        prediction = model.predict(input_data)[0]

        st.success(f"### Predicted Price: ₹ {prediction:.2f} lakh")
        st.caption(
            "Prediction generated using a Random Forest Regressor tuned via "
            "Bayesian Optimization (Optuna)."
        )

# =====================================================================
# MODEL INFO SECTION
# =====================================================================
st.divider()
st.subheader("About this model")
st.write("""
This prototype uses a Random Forest Regressor trained on the Bengaluru House
Price dataset, with hyperparameters selected via Bayesian Optimization (Optuna)
after comparing against Grid Search and Random Search strategies.
""")
