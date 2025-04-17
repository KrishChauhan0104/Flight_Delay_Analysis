import streamlit as st
import pandas as pd
import joblib

# --- Load model and scaler ---
model  = joblib.load('flight_delay_model.pkl')
scaler = joblib.load('scaler.pkl')

# Grab the exact feature list the scaler was fitted on
FEATURES = list(scaler.feature_names_in_)

st.title("✈️ Flight Delay Predictor")

# --- User inputs ---
airline        = st.number_input("Airline code", min_value=0)
origin_airport = st.number_input("Origin airport code", min_value=0)
dest_airport   = st.number_input("Destination airport code", min_value=0)
sch_hour       = st.number_input("Scheduled hour (0–23)", min_value=0, max_value=23)
dow            = st.selectbox("Day of week (0=Mon)", list(range(7)))
month          = st.selectbox("Month", list(range(1, 13)))

# --- Compute TIME_OF_DAY code exactly as in training ---
if sch_hour < 12:
    tod_label = 'Morning'
elif sch_hour < 18:
    tod_label = 'Afternoon'
else:
    tod_label = 'Evening'

# Convert that label into the same numeric code you used in training
time_of_day = pd.Series([tod_label]).astype('category').cat.codes.iloc[0]

# --- Build raw input dict & DataFrame (unindexed) ---
raw = {
    'AIRLINE': airline,
    'ORIGIN_AIRPORT': origin_airport,
    'DESTINATION_AIRPORT': dest_airport,
    'SCHEDULED_HOUR': sch_hour,
    'TIME_OF_DAY': time_of_day,
    'DAY_OF_WEEK': dow,
    'MONTH': month
}
input_df = pd.DataFrame([raw]).reindex(columns=FEATURES)

# Optional: show feature names to verify
st.write("📝 Using features:", input_df.columns.tolist())

# --- Predict on button click ---
if st.button("Predict Delay"):

    # 1) Show the input values that you’re using
    st.subheader("🔍 Input values")
    st.table(input_df)

    # 2) Scale & predict
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]

    # 3) Display result
    st.subheader("📊 Prediction")
    if pred == 1:
        st.error("🔴 Flight is likely to be delayed.")
    else:
        st.success("🟢 Flight is likely to be on time.")

