import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(color_codes=True)
df = pd.read_csv('Diamond Price Prediction.csv')
df

Exploratory Data Analysis:

sns.barplot(data=df, x="Cut(Quality)", y="Price(in US dollars)")
#Premium Quality has the highest price for diamond

sns.scatterplot(data=df, x="Carat(Weight of Daimond)", y="Price(in US dollars)")

sns.scatterplot(data=df, x="Carat(Weight of Daimond)", y="Price(in US dollars)", hue="Cut(Quality)", palette="deep")

sns.scatterplot(data=df, x="Carat(Weight of Daimond)", y="Price(in US dollars)", hue="Color", palette="deep")

sns.barplot(data=df, x="Color", y="Price(in US dollars)")

sns.barplot(data=df, x="Clarity", y="Price(in US dollars)")
#SI2 has the highest clarity price

Data Preprocessing:

df.dtypes

df['Cut(Quality)'].unique()

df['Color'].unique()

df['Clarity'].unique()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Cut(Quality)']= label_encoder.fit_transform(df['Cut(Quality)'])
df['Cut(Quality)'].unique()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Color']= label_encoder.fit_transform(df['Color'])
df['Color'].unique()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Clarity']= label_encoder.fit_transform(df['Clarity'])
df['Clarity'].unique()

# Machine Learning Model Building
X = df.drop('Price(in US dollars)', axis=1)
y = df['Price(in US dollars)']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(random_state=0)
dtree.fit(X_train, y_train)

from sklearn import metrics
import math
y_pred = dtree.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
rmse = math.sqrt(mse)

print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))
print('RMSE score is {}'.format(rmse))

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)

from sklearn import metrics
import math
y_pred = rf.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
rmse = math.sqrt(mse)

print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))
print('RMSE score is {}'.format(rmse))

