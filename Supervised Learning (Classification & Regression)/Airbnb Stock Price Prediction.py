The goal of this project is to predict Airbnb's stock closing prices using historical stock market data. By leveraging machine learning models, we aim to develop an effective forecasting system to assist investors and analysts in decision-making.

# Data Loading and Preprocessing
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import math

sns.set_theme(color_codes=True)

import pandas as pd

url = "https://raw.githubusercontent.com/tajamulkhann/Machine-Learning-Projects/main/Supervised%20Learning%20Projects/Airline%20Satisfaction%20Prediction%20using%20Classification%20Algorithms/ABNB.csv"

df = pd.read_csv(url)
df.head()

#Exploratory Data Analysis

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Closing Price', color='b')
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title("Airbnb Stock Closing Price Over Time")
plt.legend()
plt.xticks(rotation=90)
plt.show()

def create_features_datetime(df):
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    return df

df = create_features_datetime(df)
df.head()

# Machine Learning Model for Stock Price Prediction

X = df.drop(columns=['Close'])
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model I: Decision Tree Regressor

dtree = DecisionTreeRegressor(random_state=0)
dtree.fit(X_train, y_train)

y_pred_dt = dtree.predict(X_test)

mae_dt = metrics.mean_absolute_error(y_test, y_pred_dt)
mse_dt = metrics.mean_squared_error(y_test, y_pred_dt)
r2_dt = metrics.r2_score(y_test, y_pred_dt)
rmse_dt = math.sqrt(mse_dt)

print(f'Decision Tree - MAE: {mae_dt}, MSE: {mse_dt}, R2: {r2_dt}, RMSE: {rmse_dt}')

# Model II: Random Forest Regressor

rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

mae_rf = metrics.mean_absolute_error(y_test, y_pred_rf)
mse_rf = metrics.mean_squared_error(y_test, y_pred_rf)
r2_rf = metrics.r2_score(y_test, y_pred_rf)
rmse_rf = math.sqrt(mse_rf)

print(f'Random Forest - MAE: {mae_rf}, MSE: {mse_rf}, R2: {r2_rf}, RMSE: {rmse_rf}')

# Visualizing Predictions

df['Predicted_Close'] = rf.predict(X)

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Actual Close Price', color='blue')
plt.plot(df.index, df['Predicted_Close'], label='Predicted Close Price', color='red', linestyle='dashed')
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title("Airbnb Stock Price Prediction using Random Forest")
plt.legend()
plt.xticks(rotation=90)
plt.show()

