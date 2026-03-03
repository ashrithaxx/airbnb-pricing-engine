import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("airbnb_pricing_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Airbnb Dynamic Pricing Engine")

st.write("Enter listing details to predict price")

construction_year = st.number_input("Construction Year", 1900, 2025, 2015)
minimum_nights = st.number_input("Minimum Nights", 1, 30, 2)
number_of_reviews = st.number_input("Number of Reviews", 0, 1000, 50)
reviews_per_month = st.number_input("Reviews per Month", 0.0, 20.0, 3.0)
review_rate_number = st.number_input("Review Rating", 0.0, 5.0, 4.5)
host_listings = st.number_input("Host Listing Count", 1, 20, 1)
availability = st.slider("Availability (days)", 0, 365, 150)
service_fee = st.number_input("Service Fee", 0.0, 500.0, 20.0)

borough = st.selectbox("Borough", ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"])
room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
cancellation = st.selectbox("Cancellation Policy", ["flexible", "moderate", "strict"])
instant = st.selectbox("Instant Bookable", ["True", "False"])
verified = st.selectbox("Host Verified", ["verified", "unconfirmed"])

if st.button("Predict Price"):

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
    predicted_price = np.expm1(predicted_log_price)

    st.success(f"Recommended Price: ${round(predicted_price[0], 2)}")

    st.markdown("### Revenue Simulation")

    occupancy = st.slider("Estimated Occupied Nights Per Year", 0, 365, 200)

    estimated_revenue = predicted_price[0] * occupancy

    st.info(f"Estimated Annual Revenue: ${round(estimated_revenue, 2)}")