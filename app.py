import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1E3A8A;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    background-color: #2563EB;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 50px;
    border: none;
}

.stButton>button:hover {
    background-color: #1D4ED8;
    color: white;
}

.result-box {
    background-color: #DCFCE7;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #166534;
}
</style>
""", unsafe_allow_html=True)

# Load model and scaler
model = pickle.load(open("rf_model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

# Title Section
st.markdown('<p class="title">🏥 Medical Insurance Cost Predictor</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Predict your estimated insurance charges using Machine Learning</p>',
    unsafe_allow_html=True
)

# Input Fields
age = st.number_input("🎂 Enter Age", min_value=0, max_value=100, step=1)

sex = st.selectbox(
    "👤 Select Gender",
    ["male", "female"]
)

bmi = st.number_input(
    "⚖️ Enter BMI",
    min_value=10.0,
    max_value=60.0,
    step=0.1
)

children = st.number_input(
    "👶 Number of Children",
    min_value=0,
    max_value=10,
    step=1
)

smoker = st.selectbox(
    "🚬 Do you smoke?",
    ["yes", "no"]
)

region = st.selectbox(
    "🌍 Select Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Prediction Button
if st.button("Predict Insurance Cost"):

    # Create DataFrame
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Encoding
    input_df['sex'] = input_df['sex'].map({'male':1, 'female':0})

    input_df['smoker'] = input_df['smoker'].map({'yes':1, 'no':0})

    # One-hot encode region
    region_encoded = pd.get_dummies(input_df['region'], prefix='region')

# Ensure all columns exist
    for col in ['region_northeast', 'region_northwest',
            'region_southeast', 'region_southwest']:
        if col not in region_encoded:
            region_encoded[col] = 0

# Remove old region column
    input_df.drop('region', axis=1, inplace=True)

# Add encoded region columns
    input_df = pd.concat([input_df, region_encoded], axis=1)

    # Scaling
    num_cols = ['age', 'bmi', 'children']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Prediction
    prediction = model.predict(input_df)

    # Result Display
    st.markdown(
        f'''
        <div class="result-box">
            💰 Estimated Insurance Cost <br><br>
            ${prediction[0]:,.2f}
        </div>
        ''',
        unsafe_allow_html=True
    )