import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# ==============================
# Set Background
# ==============================
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    /* Target the success message text specifically */
    div[data-testid="stSuccess"] > div {{
        color: black !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("assets/background.png")

# ==============================
# Load the trained model
# ==============================
model = joblib.load("shoe_price_model.joblib")

# Load dataset again (for choices & encoding)
df = pd.read_csv("dataset.csv")

# Clean color column same way as training
color_stats = df['color'].value_counts()
rare_colors = color_stats[color_stats < 10]
df['color'] = df['color'].apply(lambda x: 'Other' if x in rare_colors.index else x)

# Drop outliers same way as training
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
df = df[(df['price'] >= lower_limit) & (df['price'] <= upper_limit)]

# One-hot encode like before
df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

# Extract feature columns used in training
feature_columns = [col for col in df_encoded.columns if col != "price"]

# ==============================
# Streamlit UI
# ==============================
st.title("👟 Shoe Price Prediction App")

st.markdown("Enter details of a shoe (Brand, Color, Size) and get the **predicted price**.")

# User inputs
brand = st.selectbox("Select Brand", sorted(df['brand'].unique()))
color = st.selectbox("Select Color", sorted(df['color'].unique()))
size = st.number_input("Enter Shoe Size", min_value=5, max_value=15, value=9)

# Button
if st.button("Predict Price"):
    # Make a single row dataframe with same columns as training
    input_dict = {
        'size': size
    }

    # Add brand/color one-hot encoding
    for col in feature_columns:
        if col.startswith("brand_"):
            input_dict[col] = 1 if col == f"brand_{brand}" else 0
        elif col.startswith("color_"):
            input_dict[col] = 1 if col == f"color_{color}" else 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure all missing columns are filled
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[feature_columns]

    # Prediction
    predicted_price = model.predict(input_df)[0]

    st.success(f"💰 Predicted Shoe Price: ₹{predicted_price:.2f}")