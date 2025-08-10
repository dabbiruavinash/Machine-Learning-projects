import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)
df = pd.read_csv('Train.csv')
df.head()

# Exploratory Data Analysis:

sns.barplot(data=df, x="Reached.on.Time_Y.N", y="Weight_in_gms")
#Heavy weight items are most likely to be late than light weight items

sns.barplot(data=df, x="Reached.on.Time_Y.N", y="Discount_offered")
#Because the huge discount offered are ontime, we have to make campaign to give more discount on product

sns.barplot(data=df, x="Reached.on.Time_Y.N", y="Cost_of_the_Product")

sns.countplot(data=df, x="Reached.on.Time_Y.N", hue="Mode_of_Shipment")

sns.countplot(data=df, x="Reached.on.Time_Y.N", hue="Gender")

sns.countplot(data=df, x="Reached.on.Time_Y.N", hue="Product_importance")

# Data Preprocessing:

df['Warehouse_block'].unique()

df['Mode_of_Shipment'].unique()

df['Product_importance'].unique()

df['Gender'].unique()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Warehouse_block']= label_encoder.fit_transform(df['Warehouse_block'])
df['Warehouse_block'].unique()

df['Mode_of_Shipment']= label_encoder.fit_transform(df['Mode_of_Shipment'])
df['Mode_of_Shipment'].unique()

df['Product_importance']= label_encoder.fit_transform(df['Product_importance'])
df['Product_importance'].unique()

df['Gender']= label_encoder.fit_transform(df['Gender'])
df['Gender'].unique()

df.head()

#convert object data types column to integer
df['Warehouse_block'] = pd.to_numeric(df['Warehouse_block'])
df['Mode_of_Shipment'] = pd.to_numeric(df['Mode_of_Shipment'])
df['Product_importance'] = pd.to_numeric(df['Product_importance'])
df['Gender'] = pd.to_numeric(df['Gender'])
df.dtypes

# Check the Outliers:

sns.boxplot(x=df["Cost_of_the_Product"])

sns.boxplot(x=df["Discount_offered"])

sns.boxplot(x=df["Weight_in_gms"])

df

# Delete the Outlier Using Z-Score:

import scipy.stats as stats
z = np.abs(stats.zscore(df))
data_clean = df[(z<3).all(axis = 1)] 
data_clean.shape

# Balance the Class Value:

#Counting 1 and 0 Value in stroke column
sns.countplot(data_clean['Reached.on.Time_Y.N'])
data_clean['Reached.on.Time_Y.N'].value_counts()

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = data_clean[(data_clean['Reached.on.Time_Y.N']==1)] 
df_minority = data_clean[(data_clean['Reached.on.Time_Y.N']==0)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 6282, # to match majority class
                                 random_state=0)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])

sns.countplot(df_upsampled['Reached.on.Time_Y.N'])
df_upsampled['Reached.on.Time_Y.N'].value_counts()

sns.heatmap(data_clean.corr(), fmt='.2g')

# Machine Learning Model Building
X = df_upsampled.drop('Reached.on.Time_Y.N', axis=1)
y = df_upsampled['Reached.on.Time_Y.N']

#test size 20% and train size 80%
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))

# Random Forest Classifier:
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))

