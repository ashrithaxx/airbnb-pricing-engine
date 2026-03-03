import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Airbnb Pricing Engine", layout="wide")

model = joblib.load("airbnb_pricing_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Airbnb Dynamic Pricing Engine")
st.markdown("Optimize listing prices using machine learning")

st.markdown("Ashritha Anandal")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Details")
    construction_year = st.number_input("Construction Year", 1900, 2025, 2015)
    minimum_nights = st.number_input("Minimum Nights", 1, 30, 2)
    availability = st.slider("Availability (days per year)", 0, 365, 150)
    service_fee = st.number_input("Service Fee", 0.0, 500.0, 20.0)

with col2:
    st.subheader("Host & Performance Metrics")
    number_of_reviews = st.number_input("Number of Reviews", 0, 1000, 50)
    reviews_per_month = st.number_input("Reviews per Month", 0.0, 20.0, 3.0)
    review_rate_number = st.number_input("Review Rating", 0.0, 5.0, 4.5)
    host_listings = st.number_input("Host Listing Count", 1, 20, 1)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    borough = st.selectbox("Borough", ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"])
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])

with col4:
    cancellation = st.selectbox("Cancellation Policy", ["flexible", "moderate", "strict"])
    instant = st.selectbox("Instant Bookable", ["True", "False"])
    verified = st.selectbox("Host Verified", ["verified", "unconfirmed"])

st.markdown("---")

occupancy = st.slider("Estimated Occupied Nights Per Year", 0, 365, 200)

st.markdown("")

if st.button("Predict Optimal Price"):

    input_data = {
        "Construction year": construction_year,
        "minimum nights": minimum_nights,
        "number of reviews": number_of_reviews,
        "reviews per month": reviews_per_month,
        "review rate number": review_rate_number,
        "calculated host listings count": host_listings,
        "availability 365": availability,
        "service fee": service_fee
    }

    input_df = pd.DataFrame([input_data])

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    for col in model_columns:
        if borough in col:
            input_df[col] = 1
        if room_type in col:
            input_df[col] = 1
        if cancellation in col:
            input_df[col] = 1
        if instant in col:
            input_df[col] = 1
        if verified in col:
            input_df[col] = 1

    input_df = input_df[model_columns]

    predicted_log_price = model.predict(input_df)
    predicted_price = np.expm1(predicted_log_price)[0]

    estimated_revenue = predicted_price * occupancy

    st.markdown("---")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            label="Recommended Nightly Price",
            value=f"${round(predicted_price, 2)}"
        )

    with result_col2:
        st.metric(
            label="Estimated Annual Revenue",
            value=f"${round(estimated_revenue, 2)}"
        )

    st.markdown("")
    st.success("Pricing recommendation generated successfully.")

