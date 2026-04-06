import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_gold_data
from model import create_dataset, build_model
from utils import predict_future

# ---------------------------
# UI CONFIG
# ---------------------------
st.set_page_config(page_title="XAUUSD Predictor", layout="wide")

st.title("🥇 Gold Price Prediction (XAUUSD) using LSTM")
st.write("Forecast future gold prices using deep learning (LSTM).")

# ---------------------------
# LOAD DATA
# ---------------------------
df = load_gold_data()

# 🔥 SAFETY FIX (for MultiIndex issue)
import pandas as pd
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Ensure only Close column is used
df = df[['Close']]

# ---------------------------
# DISPLAY DATA
# ---------------------------
st.subheader("📊 Historical Gold Prices")

# ✅ FIXED (no error now)
st.line_chart(df['Close'])

# Convert to numpy
data = df.values.astype("float32")

# ---------------------------
# PREPROCESSING
# ---------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("⚙️ Settings")

look_back = st.sidebar.slider("Look Back Window", 1, 10, 5)
epochs = st.sidebar.slider("Epochs", 10, 100, 30)
future_days = st.sidebar.slider("Forecast Days", 1, 30, 30)

# ---------------------------
# PREPARE DATA
# ---------------------------
X, Y = create_dataset(data_scaled, look_back)

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# ---------------------------
# TRAIN MODEL
# ---------------------------
if st.button("🚀 Train Model"):

    with st.spinner("Training LSTM model..."):
        model = build_model(look_back)
        model.fit(X, Y, epochs=epochs, batch_size=32, verbose=0)

    st.success("✅ Model Trained Successfully!")

    # ---------------------------
    # PREDICTIONS
    # ---------------------------
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # ---------------------------
    # PLOT: ACTUAL vs PREDICTED
    # ---------------------------
    st.subheader("📈 Actual vs Predicted")

    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(df['Close'].values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Model Performance")

    st.pyplot(fig1)

    # ---------------------------
    # FUTURE FORECAST
    # ---------------------------
    future_preds = predict_future(
        model, data_scaled, scaler,
        days=future_days,
        look_back=look_back
    )

    st.subheader(f"🔮 Next {future_days} Days Forecast")

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(range(len(df)), df['Close'].values, label="Historical")

    plt.plot(
        range(len(df), len(df) + future_days),
        future_preds,
        label="Forecast"
    )

    plt.legend()
    plt.title("Future Prediction")

    st.pyplot(fig2)

    # ---------------------------
    # SHOW LAST VALUES (PRO TOUCH)
    # ---------------------------
    st.subheader("📌 Latest Data")

    st.write("Last Actual Price:", df['Close'].iloc[-1])
    st.write("Next Predicted Price:", future_preds[0][0])