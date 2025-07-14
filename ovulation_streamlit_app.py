#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go

# Load model
@st.cache_resource
def load_model():
    return joblib.load("ovulation_model.pkl")

model = load_model()

# Get desktop path
def get_desktop_path():
    return os.path.join(os.path.expanduser("~"), "Desktop")

# Predict function
def predict_ovulation(inputs, last_period):
    start_date = datetime.strptime(last_period, "%Y-%m-%d")
    df = pd.DataFrame([inputs], columns=['age', 'cycle_length', 'luteal_phase', 'avg_bbt', 'mucus_score', 'mood_score'])
    predicted_day = round(model.predict(df)[0])
    ovulation_date = start_date + timedelta(days=predicted_day - 1)
    fertile_window = [ovulation_date + timedelta(days=i) for i in [-2, -1, 0, 1]]
    return ovulation_date, fertile_window

# Save prediction
def save_prediction(record):
    csv_file = "ovulation_prediction_history.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(csv_file, index=False)

# UI
st.set_page_config("Ovulation Predictor", layout="centered")
st.title("🌸 Intelligent Ovulation Predictor")

with st.form("ovulation_form"):
    age = st.number_input("Age", min_value=10, max_value=55, value=28)
    cycle_length = st.number_input("Cycle Length", min_value=20, max_value=40, value=28)
    luteal_phase = st.number_input("Luteal Phase", min_value=10, max_value=20, value=14)
    avg_bbt = st.number_input("Average BBT (°C)", min_value=35.0, max_value=38.0, value=36.5)
    mucus_score = st.slider("Mucus Score (1-5)", 1, 5, 3)
    mood_score = st.slider("Mood Score (1-5)", 1, 5, 3)
    last_period = st.date_input("Last Period Start Date")
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    inputs = [age, cycle_length, luteal_phase, avg_bbt, mucus_score, mood_score]
    ov_date, fertile_days = predict_ovulation(inputs, last_period.strftime("%Y-%m-%d"))

    st.success(f"🎯 **Ovulation Day**: {ov_date.date()}")
    st.info(f"🌱 **Fertile Window**: {fertile_days[0].date()} to {fertile_days[-1].date()}")

    # Save
    record = {
        'date_predicted': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'age': age,
        'cycle_length': cycle_length,
        'luteal_phase': luteal_phase,
        'avg_bbt': avg_bbt,
        'mucus_score': mucus_score,
        'mood_score': mood_score,
        'last_period': last_period.strftime('%Y-%m-%d'),
        'predicted_ovulation': ov_date.strftime('%Y-%m-%d'),
        'fertile_window_start': fertile_days[0].strftime('%Y-%m-%d'),
        'fertile_window_end': fertile_days[-1].strftime('%Y-%m-%d')
    }
    save_prediction(record)

# Show history
st.subheader("📊 Prediction History")
if os.path.exists("ovulation_prediction_history.csv"):
    df = pd.read_csv("ovulation_prediction_history.csv")
    st.dataframe(df)

    # Trend Chart
    st.subheader("📈 Ovulation Trend Chart")
    df['date_predicted'] = pd.to_datetime(df['date_predicted'])
    df['predicted_ovulation_day'] = pd.to_datetime(df['predicted_ovulation'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date_predicted'], y=df['predicted_ovulation_day'],
                             mode='lines+markers', name='Ovulation Prediction'))
    fig.update_layout(title="Ovulation Date Trend",
                      xaxis_title="Date Predicted",
                      yaxis_title="Predicted Ovulation Date",
                      height=400)
    st.plotly_chart(fig)

    # Export CSV
    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download History as CSV", csv_export, "ovulation_history.csv", "text/csv")

else:
    st.warning("No history found yet. Make a prediction first.")


