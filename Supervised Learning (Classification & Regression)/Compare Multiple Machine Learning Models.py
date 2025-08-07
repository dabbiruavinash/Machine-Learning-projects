By comparing multiple models, we aim to select the most effective algorithm that offers the optimal balance of accuracy, complexity, and performance for their specific problem. Below is the process we can follow for the task of comparing multiple Machine Learning models:

Address missing values, remove duplicates, and correct errors in the dataset to ensure the quality of data fed into the models.
Divide the dataset into training and testing sets, typically using a 70-30% or 80-20% split.
Select a diverse set of models for comparison. It can include simple linear models, tree-based models, ensemble methods, and more advanced algorithms, depending on the problem’s complexity and data characteristics.
Fit each selected model to the training data. It involves adjusting the model to learn from the features and the target variable in the training set.
Use a set of metrics to evaluate each model’s performance on the test set.
Compare the models based on the evaluation metrics, considering both their performance and computational efficiency.

import pandas as pd
data = pd.read_csv('Real_Estate.csv')

# display the first few rows
data_head = data.head()

print(data_head)

print(data.info())

The dataset consists of 414 entries and 7 columns, with no missing values. Here’s a brief overview of the columns:

Transaction date: The date of the house sale (object type, which suggests it might need conversion or extraction of useful features like year, month, etc.).
House age: The age of the house in years (float).
Distance to the nearest MRT station: The distance to the nearest mass rapid transit station in meters (float).
Number of convenience stores: The number of convenience stores in the living circle on foot (integer).
Latitude: The geographic coordinate that specifies the north-south position (float).
Longitude: The geographic coordinate that specifies the east-west position (float).
House price of unit area: Price of the house per unit area (float), which is likely our target variable for prediction.

Data Preprocessing:

Let’s start with the preprocessing steps. Below are the steps we will follow to preprocess our data:

Since the transaction date is in a string format, we will convert it into a datetime object. We can then extract features such as the transaction year and month, which might be useful for the model.
We’ll scale the continuous features to ensure they’re on a similar scale. It is particularly important for models like Support Vector Machines or K-nearest neighbours, which are sensitive to the scale of input features.
We’ll split the dataset into a training set and a testing set. A common practice is to use 80% of the data for training and 20% for testing.

Let’s implement these preprocessing steps:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime

# convert "Transaction date" to datetime and extract year and month
data['Transaction date'] = pd.to_datetime(data['Transaction date'])
data['Transaction year'] = data['Transaction date'].dt.year
data['Transaction month'] = data['Transaction date'].dt.month

# drop the original "Transaction date" as we've extracted relevant features
data = data.drop(columns=['Transaction date'])

# define features and target variable
X = data.drop('House price of unit area', axis=1)
y = data['House price of unit area']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape

Model Training and Comparison:

Linear Regression: A good baseline model for regression tasks.
Decision Tree Regressor: To see how a simple tree-based model performs.
Random Forest Regressor: An ensemble method to improve upon the decision tree’s performance.
Gradient Boosting Regressor: Another powerful ensemble method for regression.

We’ll train each model using the training data and evaluate their performance on the test set using Mean Absolute Error (MAE) and R-squared (R²) as metrics. These metrics will help us understand both the average error of the predictions and how well the model explains the variance in the target variable.

Let’s start with training these models and comparing their performance:

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# initialize the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# dictionary to hold the evaluation metrics for each model
results = {}

# train and evaluate each model
for name, model in models.items():
    # training the model
    model.fit(X_train_scaled, y_train)

    # making predictions on the test set
    predictions = model.predict(X_test_scaled)

    # calculating evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # storing the metrics
    results[name] = {"MAE": mae, "R²": r2}

results_df = pd.DataFrame(results).T  # convert the results to a DataFrame for better readability
print(results_df)

Linear Regression has the lowest MAE (9.75) and the highest R² (0.53), making it the best-performing model among those evaluated. It suggests that, despite its simplicity, Linear Regression is quite effective for this dataset.

Decision Tree Regressor shows the highest MAE (11.76) and the lowest R² (0.20), indicating it may be overfitting to the training data and performing poorly on the test data. On the other hand, Random Forest Regressor and Gradient Boosting Regressor have similar MAEs (9.89 and 10.00, respectively) and R² scores (0.51 and 0.48, respectively), performing slightly worse than the Linear Regression model but better than the Decision Tree.

