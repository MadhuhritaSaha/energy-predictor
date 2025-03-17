import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# Load models
lstm_model = load_model('lstm_model.h5')
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('xgb_model.json')

# Streamlit UI
st.title("🔋 Energy Consumption Predictor")
st.write("Upload your energy dataset and get predictions using a hybrid LSTM + XGBoost model.")

uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Data Preview:", df.head())

    # Preprocessing
    df['HVACUsage'] = df['HVACUsage'].map({'On': 1, 'Off': 0})
    df['LightingUsage'] = df['LightingUsage'].map({'On': 1, 'Off': 0})
    df['Holiday'] = df['Holiday'].map({'Yes': 1, 'No': 0})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    features = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
                'HVACUsage', 'LightingUsage', 'RenewableEnergy', 'Holiday',
                'Hour', 'Day', 'Month', 'DayOfWeek']
    
    target = 'EnergyConsumption'
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features + [target]])

    # Create sequence for LSTM (last 24 hours)
    seq_len = 24
    if len(df) >= seq_len:
        last_seq = scaled_data[-seq_len:, :-1]
        X_seq = np.array([last_seq])

        # LSTM prediction
        lstm_pred_scaled = lstm_model.predict(X_seq)
        temp_array = np.zeros((1, len(features)))
        combined = np.hstack((temp_array, lstm_pred_scaled))
        lstm_pred = scaler.inverse_transform(combined)[:, -1][0]

        # XGBoost prediction
        xgb_input = scaled_data[-1, :-1].reshape(1, -1)
        xgb_pred = xgb_model.predict(xgb_input)[0]

        # Final hybrid prediction
        final_pred = lstm_pred + xgb_pred

        st.success(f"⚡ Predicted Energy Consumption: **{final_pred:.2f} units**")
    else:
        st.warning("Not enough data rows for prediction (need at least 24).")
