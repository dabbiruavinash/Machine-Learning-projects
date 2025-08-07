import pandas as pd
data = pd.read_csv('/content/apple_stock_data.csv')
print(data.head())

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Close']]

So, let’s scale the Close price data between 0 and 1 using MinMaxScaler to ensure compatibility with the LSTM model:

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])

import numpy as np
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(data['Close'].values, seq_length)

Now, we will split the sequences into training and test sets (e.g., 80% training, 20% testing):

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

Now, we will build a sequential LSTM model with layers to capture the temporal dependencies in the data:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
​
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

Now, we will compile the model using an appropriate optimizer and loss function, and fit it into the training data:

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32)

Now, let’s train the second model. I’ll start by generating lagged features for Linear Regression (e.g., using the past 3 days as predictors):

data['Lag_1'] = data['Close'].shift(1)
data['Lag_2'] = data['Close'].shift(2)
data['Lag_3'] = data['Close'].shift(3)
data = data.dropna()

Now, we will split the data accordingly for training and testing:

X_lin = data[['Lag_1', 'Lag_2', 'Lag_3']]
y_lin = data['Close']
X_train_lin, X_test_lin = X_lin[:train_size], X_lin[train_size:]
y_train_lin, y_test_lin = y_lin[:train_size], y_lin[train_size:]

Now, let’s train the linear regression model:

from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train_lin, y_train_lin)

Now, here’s how to make predictions using LSTM on the test set and inverse transform the scaled predictions:

X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

Here’s how to generate predictions using Linear Regression and inverse-transform them:

lin_predictions = lin_model.predict(X_test_lin)
lin_predictions = scaler.inverse_transform(lin_predictions.reshape(-1, 1))

And, here’s how to use a weighted average to create hybrid predictions:

hybrid_predictions = (0.7 * lstm_predictions) + (0.3 * lin_predictions)

Predicting using the Hybrid Model

lstm_future_predictions = []
last_sequence = X[-1].reshape(1, seq_length, 1)
for _ in range(10):
    lstm_pred = lstm_model.predict(last_sequence)[0, 0]
    lstm_future_predictions.append(lstm_pred)
    lstm_pred_reshaped = np.array([[lstm_pred]]).reshape(1, 1, 1)
    last_sequence = np.append(last_sequence[:, 1:, :], lstm_pred_reshaped, axis=1)
lstm_future_predictions = scaler.inverse_transform(np.array(lstm_future_predictions).reshape(-1, 1))

Here’s how to predict the Next 10 Days using Linear Regression:

recent_data = data['Close'].values[-3:]
lin_future_predictions = []
for _ in range(10):
    lin_pred = lin_model.predict(recent_data.reshape(1, -1))[0]
    lin_future_predictions.append(lin_pred)
    recent_data = np.append(recent_data[1:], lin_pred)
lin_future_predictions = scaler.inverse_transform(np.array(lin_future_predictions).reshape(-1, 1))

And, here’s how to combine the predictive power of both models to make predictions for the next 10 days:

hybrid_future_predictions = (0.7 * lstm_future_predictions) + (0.3 * lin_future_predictions)

Here’s how to create the final DataFrame to look at the predictions:

future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10)
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'LSTM Predictions': lstm_future_predictions.flatten(),
    'Linear Regression Predictions': lin_future_predictions.flatten(),
    'Hybrid Model Predictions': hybrid_future_predictions.flatten()})
print(predictions_df)

