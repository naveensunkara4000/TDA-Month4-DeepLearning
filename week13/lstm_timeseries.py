import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "../data")
MODELDIR = os.path.join(BASEDIR, "../models")
OUTDIR = os.path.join(BASEDIR, "../outputs/week13")

os.makedirs(DATADIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

DATA_CSV = os.path.join(DATADIR, "stock_prices.csv")

def load_or_create_series():
    if os.path.exists(DATA_CSV):
        print("Loading real time series from:", DATA_CSV)
        df = pd.read_csv(DATA_CSV)
        # assume column "Close"
        series = df["Close"].values.astype("float32")
    else:
        print("No stock_prices.csv found. Creating synthetic sine wave series...")
        t = np.arange(0, 500)
        series = 50 + 5 * np.sin(0.05 * t) + np.random.normal(scale=0.5, size=len(t))
    return series

def create_sequences(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        # data shape: (n, 1) -> slice: (window_size, 1)
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    X = np.array(X)  # shape: (samples, window_size, 1)
    y = np.array(y)  # shape: (samples, 1)
    # Already correct shape for LSTM: (batch, timesteps, features)
    return X, y



def build_lstm(input_shape):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=False, input_shape=input_shape),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


def plot_predictions(true_values, predicted_values, title, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(true_values, label="True")
    plt.plot(predicted_values, label="Predicted")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename))
    plt.close()

def main():
    # 1. Load or create series
    series = load_or_create_series()
    series = series.reshape(-1, 1)

    # 2. Scale to [0,1]
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)
    joblib.dump(scaler, os.path.join(MODELDIR, "lstm_scaler.pkl"))

    # 3. Create sequences
    window_size = 20
    X, y = create_sequences(series_scaled, window_size=window_size)

    # Train/test split (last 20% as test)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    # 4. Build model
    model = build_lstm(input_shape=X_train.shape[1:])

    # 5. Train
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # 6. Evaluate
    y_pred_scaled = model.predict(X_test)
    # invert scaling
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled).flatten()

    mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.4f}")

    # 7. Save model
    model_path = os.path.join(MODELDIR, "lstm_timeseries.h5")
    model.save(model_path)
    print("LSTM model saved at:", model_path)

    # 8. Save metrics and history
    with open(os.path.join(OUTDIR, "evaluation.txt"), "w") as f:
        f.write(f"Test RMSE: {rmse:.4f}\n")

    # 9. Plot loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "loss.png"))
    plt.close()

    # 10. Plot predictions vs true
    plot_predictions(
        true_values=y_test_unscaled,
        predicted_values=y_pred_unscaled,
        title="LSTM Time Series Prediction",
        filename="predictions.png"
    )

    print("Week 13 done! Check outputs in:", OUTDIR)

if __name__ == "__main__":
    main()
