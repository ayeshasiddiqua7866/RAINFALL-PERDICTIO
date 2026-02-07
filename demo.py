import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Load dataset
def load_data(path):
    data = pd.read_csv(path)
    data = data[['SUBDIVISION', 'YEAR', 'ANNUAL']]
    data_pivot = data.pivot(index='YEAR', columns='SUBDIVISION', values='ANNUAL')
    return data_pivot

# Preprocess data
def preprocess_data(data, window_size):
    data = data.interpolate(method='linear', axis=0).fillna(method='ffill').fillna(method='bfill')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - window_size):
        X.append(data_scaled[i:i + window_size])
        y.append(data_scaled[i + window_size])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Define CNN-LSTM model
def create_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        LSTM(50, activation='relu', return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(input_shape[1])  # Output shape matches number of subdivisions
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Forecasting function
def forecast(model, data, steps, window_size):
    predictions = []
    input_seq = data[-window_size:].reshape(1, window_size, data.shape[1])
    for _ in range(steps):
        pred = model.predict(input_seq)[0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, -1), axis=1)
    return np.array(predictions)

# Main execution
def process(file_path,forecast_years, state_name):
    #file_path = "data.csv"  # Update with your file path
    window_size = 10
    
    # User input for forecast years and state selection
    
    
    data = load_data(file_path)
    if state_name not in data.columns:
        print(f"Error: {state_name} not found in dataset.")
        exit()
    
    X, y, scaler = preprocess_data(data, window_size)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = create_cnn_lstm((window_size, X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=1)
    
    predictions = forecast(model, scaler.transform(data), forecast_years, window_size)
    predictions = scaler.inverse_transform(predictions)
    
    future_years = range(data.index[-1] + 1, data.index[-1] + 1 + forecast_years)
    forecast_df = pd.DataFrame(predictions, index=future_years, columns=data.columns)
    
    # Show results for the selected state
    print(f"Forecasted rainfall for {state_name}:")
    print(forecast_df[[state_name]])
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-20:], data[state_name].values[-20:], label=f"Actual {state_name}")
    plt.plot(future_years, forecast_df[state_name].values, linestyle='dashed', label=f"Predicted {state_name}")
    plt.xlabel("Year")
    plt.ylabel("Annual Rainfall")
    plt.title(f"Annual Rainfall Forecasting for {state_name} using CNN-LSTM")
    plt.legend()
    plt.show()
process("data.csv",20,"ANDAMAN & NICOBAR ISLANDS")
