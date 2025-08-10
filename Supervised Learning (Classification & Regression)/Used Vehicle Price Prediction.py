import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)
df = pd.read_csv('used_vehicle.csv')
df.head()

# Data Preprocessing Part 1:

df.drop(columns=['Unnamed: 0'], inplace=True)
df.shape

df.dtypes

# Remove rows with missing values in 'City' and 'Highway' columns in the original dataframe
df.dropna(subset=[' City', ' Highway'], inplace=True)
# Extract all the characters before the first 'L' only if 'L' is present
mask = df[' City'].str.contains('L')
df.loc[mask, ' City'] = df.loc[mask, ' City'].str.split('L', n=1, expand=True)[0]

mask = df[' Highway'].str.contains('L')
df.loc[mask, ' Highway'] = df.loc[mask, ' Highway'].str.split('L', n=1, expand=True)[0]

df.head()

# Remove non-numeric characters from the 'City' and 'Highway' columns
df[' City'] = df[' City'].str.replace('[^0-9]', '').astype(int)
df[' Highway'] = df[' Highway'].str.replace('[^0-9]', '').astype(int)

print(df.dtypes)

df.shape

df.dropna(subset=['Kilometres'], inplace=True)
df['Kilometres'] = df['Kilometres'].str.replace('km', '').astype(int)
df.dtypes

df.shape

df.head()

# Add decimal point to Highway and City columns
df[' Highway'] = df[' Highway'] / 10
df[' City'] = df[' City'] / 10
df.head()

# iterate over columns with object datatype
for col in df.select_dtypes(include='object'):
    # count the unique number of values
    unique_count = df[col].nunique()
    # print the result
    print(f'The number of unique values in the "{col}" column is: {unique_count}')

def fetch_model(text):
    if 'MDX' in text:
        return 'MDX'
    elif 'TSX' in text:
        return 'TSX'
    elif 'Grand' in text:
        return 'Grand'
    elif 'Civic' in text:
        return 'Civic'
    elif 'RDX' in text:
        return 'RDX'
    elif 'ILX' in text:
        return 'ILX'
    elif 'TLX' in text:
        return 'TLX'
    else:
        return 'Other Model'

df['Model'] = df['Model'].apply(fetch_model)
plt.figure(figsize=(10,5))
df['Model'].value_counts().plot(kind='bar')

df.drop(columns=[' Exterior Colour', ' Interior Colour'], inplace=True)
df.shape

df.Make.unique()

# Make the segmentation for car brands:

def segment_make(make):
    if make in ['Acura', 'Alfa Romeo', 'Audi', 'Bentley', 'BMW', 'Cadillac', 'Genesis', 'Infiniti', 'Jaguar', 'Lamborghini', 'Land Rover', 'Lexus', 'Lincoln', 'Maserati', 'McLaren', 'Mercedes-Benz', 'Porsche', 'Rolls-Royce', 'Tesla']:
        return 'Luxury'
    elif make in ['Buick', 'Chevrolet', 'Chrysler', 'Dodge', 'Ford', 'GMC', 'Jeep', 'Ram']:
        return 'Mainstream'
    elif make in ['Ferrari', 'Lotus']:
        return 'Sports'
    elif make in ['Honda', 'Hyundai', 'Kia', 'Mazda', 'Mitsubishi', 'Nissan', 'Subaru', 'Toyota', 'Volkswagen']:
        return 'Value'
    else:
        return 'Other'

df['Make'] = df['Make'].apply(segment_make)
plt.figure(figsize=(10,5))
df['Make'].value_counts().plot(kind='bar')

# Segment body type:

df['Body Type'].unique()

# Define the body type segments
suv = ['SUV']
sedan = ['Sedan', 'Coupe', 'Convertible']
hatchback = ['Hatchback']
wagon = ['Wagon', 'Station Wagon']
truck = ['Truck', 'Truck Extended Cab', 'Extended Cab', 'Crew Cab',
         'Regular Cab', 'Truck Crew Cab', 'Super Cab', 'Quad Cab',
         'Truck Super Cab', 'Truck Double Cab', 'Truck King Cab',
         'Truck Long Crew Cab']
van = ['Van Regular', 'Van Extended']
minivan = ['Minivan']
roadster = ['Roadster']
cabriolet = ['Cabriolet']
super_crew = ['Super Crew']
compact = ['Compact']

# Create a dictionary to map each body type to its corresponding segment
body_type_segments = {}
for body_type in df['Body Type'].unique():
    if body_type in suv:
        body_type_segments[body_type] = 'SUV'
    elif body_type in sedan:
        body_type_segments[body_type] = 'Sedan'
    elif body_type in hatchback:
        body_type_segments[body_type] = 'Hatchback'
    elif body_type in wagon:
        body_type_segments[body_type] = 'Wagon'
    elif body_type in truck:
        body_type_segments[body_type] = 'Truck'
    elif body_type in van:
        body_type_segments[body_type] = 'Van'
    elif body_type in minivan:
        body_type_segments[body_type] = 'Minivan'
    elif body_type in roadster:
        body_type_segments[body_type] = 'Roadster'
    elif body_type in cabriolet:
        body_type_segments[body_type] = 'Cabriolet'
    elif body_type in super_crew:
        body_type_segments[body_type] = 'Super Crew'
    elif body_type in compact:
        body_type_segments[body_type] = 'Compact'
    else:
        body_type_segments[body_type] = 'Other'

# Map the body type segments to the dataframe
df['Body Type'] = df['Body Type'].map(body_type_segments)

plt.figure(figsize=(10,5))
df['Body Type'].value_counts().plot(kind='bar')

# Segment Transmission:

df[' Transmission'].unique()

def segment_transmission(transmission):
    if transmission in ['Automatic', 'CVT', '1 Speed Automatic']:
        return 'Automatic'
    elif transmission in ['6 Speed Manual', '5 Speed Manual', '7 Speed Manual']:
        return 'Manual'
    elif transmission in ['9 Speed Automatic', '10 Speed Automatic', '8 Speed Automatic', '7 Speed Automatic', '5 Speed Automatic', '4 Speed Automatic']:
        return 'Traditional Automatic'
    elif transmission in ['8 Speed Automatic with auto-shift', '6 Speed Automatic with auto-shift', '7 Speed Automatic with auto-shift', '5 Speed Automatic with auto-shift']:
        return 'Automated Manual'
    elif transmission == 'Sequential':
        return 'Semi-Automatic'
    elif transmission == 'F1 Transmission':
        return 'Automated Single-Clutch'
    else:
        return 'Unknown'
    
df['Transmission'] = df[' Transmission'].apply(segment_transmission)

plt.figure(figsize=(10,5))
df['Transmission'].value_counts().plot(kind='bar')

df.drop(columns=[' Transmission'], inplace=True)
df.shape

# Drop Engine column
df.drop(columns=[' Engine'], inplace=True)
df.shape

df.head()

# Exploratory Data Analysis
# iterate over columns with object datatype
for col in df.select_dtypes(include='object'):
    # count the unique number of values
    unique_count = df[col].nunique()
    # print the result
    print(f'The number of unique values in the "{col}" column is: {unique_count}')

# list of categorical variables to plot
cat_vars = ['Make', 'Model', 'Body Type', ' Drivetrain', ' Doors', ' Fuel Type', 'Transmission']

# create figure with subplots
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axs = axs.ravel()

# create barplot for each categorical variable
for i, var in enumerate(cat_vars):
    sns.barplot(x=var, y='Price', data=df, ax=axs[i], estimator=np.mean)
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

# remove the eighth subplot
fig.delaxes(axs[7])

# adjust spacing between subplots
fig.tight_layout()

# show plot
plt.show()

num_vars = [' Passengers', ' City', ' Highway']

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.boxplot(x=var, data=df, ax=axs[i])

fig.tight_layout()

plt.show()

num_vars = [' Passengers', ' City', ' Highway']

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.boxplot(x=var, data=df, ax=axs[i])

fig.tight_layout()

plt.show()

num_vars = [' Passengers', ' City', ' Highway']

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.scatterplot(x=var, y='Price', hue='Make', data=df, ax=axs[i])

fig.tight_layout()

plt.show()


sns.set_style("darkgrid")
sns.set_palette("Set2")

sns.lineplot(x='Year', y='Price', hue='Make', data=df, ci=None, estimator='mean', alpha=0.7)

plt.title("Price by Year Sold and Car Type")
plt.xlabel("Year")
plt.ylabel("Price")

plt.show()

# Data Preprocessing Part 2
check_missing = df.isnull().sum() * 100 / df.shape[0]
check_missing[check_missing > 0].sort_values(ascending=False)

df.drop(columns=[' Passengers'], inplace=True)
df[' Doors'] = df[' Doors'].fillna(df[' Doors'].median())
df.dropna(subset=[' Drivetrain'], inplace=True)
df.shape

# Label Encoding for Object datatype
# Loop over each column in the DataFrame where dtype is 'object'
for col in df.select_dtypes(include=['object']).columns:
    
    # Print the column name and the unique values
    print(f"{col}: {df[col].unique()}")

df[' Doors'] = df[' Doors'].astype(float)
from sklearn import preprocessing

# Loop over each column in the DataFrame where dtype is 'object'
for col in df.select_dtypes(include=['object']).columns:
    
    # Initialize a LabelEncoder object
    label_encoder = preprocessing.LabelEncoder()
    
    # Fit the encoder to the unique values in the column
    label_encoder.fit(df[col].unique())
    
    # Transform the column using the encoder
    df[col] = label_encoder.transform(df[col])
    
    # Print the column name and the unique encoded values
    print(f"{col}: {df[col].unique()}")

df.dtypes

# Remove Outlier using Z-Score
from scipy import stats

# define a function to remove outliers using z-score for only selected numerical columns
def remove_outliers(df, cols, threshold=3):
    # loop over each selected column
    for col in cols:
        # calculate z-score for each data point in selected column
        z = np.abs(stats.zscore(df[col]))
        # remove rows with z-score greater than threshold in selected column
        df = df[(z < threshold) | (df[col].isnull())]
    return df

selected_cols = [' City', ' Highway']
df_clean = remove_outliers(df, selected_cols)
df_clean.shape

#dataframe before the outlier removed
df.shape

#Correlation Heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(df_clean.corr(), fmt='.2g', annot=True)

# Machine Learning Model Building
X = df_clean.drop('Price', axis=1)
y = df_clean['Price']

#test size 20% and train size 80%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston


# Create a DecisionTreeRegressor object
dtree = DecisionTreeRegressor()

# Define the hyperparameters to tune and their values
param_grid = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(grid_search.best_params_)

from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(random_state=0, max_depth=8, max_features='auto', min_samples_leaf=4, min_samples_split=8)
dtree.fit(X_train, y_train)

from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
import math
y_pred = dtree.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
rmse = math.sqrt(mse)

print('MAE is {}'.format(mae))
print('MAPE is {}'.format(mape))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))
print('RMSE score is {}'.format(rmse))

imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": dtree.feature_importances_
})
fi = imp_df.sort_values(by="Importance", ascending=False)

fi2 = fi.head(10)
plt.figure(figsize=(10,8))
sns.barplot(data=fi2, x='Importance', y='Feature Name')
plt.title('Feature Importance Each Attributes (Decision Tree Regressor)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

import shap
explainer = shap.TreeExplainer(dtree)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

explainer = shap.Explainer(dtree, X_test)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Create a Random Forest Regressor object
rf = RandomForestRegressor()

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0, max_depth=9, min_samples_split=2, min_samples_leaf=1, 
                           max_features='sqrt')
rf.fit(X_train, y_train)

from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
import math
y_pred = rf.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
rmse = math.sqrt(mse)

print('MAE is {}'.format(mae))
print('MAPE is {}'.format(mape))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))
print('RMSE score is {}'.format(rmse))

imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": dtree.feature_importances_
})
fi = imp_df.sort_values(by="Importance", ascending=False)

fi2 = fi.head(10)
plt.figure(figsize=(10,8))
sns.barplot(data=fi2, x='Importance', y='Feature Name')
plt.title('Feature Importance Each Attributes (Random Forest Regressor)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

explainer = shap.Explainer(rf, X_test, check_additivity=False)
shap_values = explainer(X_test, check_additivity=False)
shap.plots.waterfall(shap_values[0])

# AdaBoost Regressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

# Define AdaBoostRegressor model
abr = AdaBoostRegressor()

# Define hyperparameters and possible values
params = {'n_estimators': [50, 100, 150],
          'learning_rate': [0.01, 0.1, 1, 10]}

# Perform GridSearchCV with 5-fold cross validation
grid_search = GridSearchCV(abr, param_grid=params, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print best hyperparameters and corresponding score
print("Best hyperparameters: ", grid_search.best_params_)

from sklearn.ensemble import RandomForestRegressor
abr = AdaBoostRegressor(random_state=0, learning_rate=0.1, n_estimators=50)
abr.fit(X_train, y_train)

from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
import math
y_pred = abr.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
rmse = math.sqrt(mse)

print('MAE is {}'.format(mae))
print('MAPE is {}'.format(mape))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))
print('RMSE score is {}'.format(rmse))

imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": abr.feature_importances_
})
fi = imp_df.sort_values(by="Importance", ascending=False)

fi2 = fi.head(10)
plt.figure(figsize=(10,8))
sns.barplot(data=fi2, x='Importance', y='Feature Name')
plt.title('Feature Importance Each Attributes (AdaBoost Regressor)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

