import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(color_codes=True)
     
df = pd.read_csv('drug200.csv')
df

# Exploratory Data Analysis:

sns.barplot(data=df, x="Cholesterol",y="Na_to_K", hue="Drug")

sns.countplot(data=df, x="Drug", hue="Sex")

sns.countplot(data=df, x="BP", hue="Drug")

sns.barplot(data=df, x="Drug", y="Age")

# Feature Engineering:

df['Sex'].unique()

df['BP'].unique()

df['Cholesterol'].unique()

df['Drug'].unique()

#Change value in Sex column
df['Sex'] = df['Sex'].replace(['F'],'0')
df['Sex'] = df['Sex'].replace(['M'],'1')
#Change value in BP column
df['BP'] = df['BP'].replace(['LOW'],'0')
df['BP'] = df['BP'].replace(['NORMAL'],'1')
df['BP'] = df['BP'].replace(['HIGH'],'2')
#Change value in Cholesterol column
df['Cholesterol'] = df['Cholesterol'].replace(['NORMAL'],'0')
df['Cholesterol'] = df['Cholesterol'].replace(['HIGH'],'1')
#Change value in Drug column
df['Drug'] = df['Drug'].replace(['DrugY'],'0')
df['Drug'] = df['Drug'].replace(['drugC'],'1')
df['Drug'] = df['Drug'].replace(['drugX'],'2')
df['Drug'] = df['Drug'].replace(['drugA'],'3')
df['Drug'] = df['Drug'].replace(['drugB'],'4')
df.head()

# Change the Datatype:

#convert object data types column to integer
df['Sex'] = pd.to_numeric(df['Sex'])
df['BP'] = pd.to_numeric(df['BP'])
df['Cholesterol'] = pd.to_numeric(df['Cholesterol'])
df['Drug'] = pd.to_numeric(df['Drug'])
df.dtypes

# Check the Outlier:

sns.boxplot(x=df["Age"])

sns.boxplot(x=df["Na_to_K"])
     
# Remove the Outlier:

import scipy.stats as stats
import numpy as np
z = np.abs(stats.zscore(df))
data_clean = df[(z<3).all(axis = 1)] 
data_clean.shape
     
# Print the Outlier:

df[~df.index.isin(data_clean.index)]

# Data Correlation

sns.heatmap(data_clean.corr(), fmt='.2g')

corr = data_clean[data_clean.columns[1:]].corr()['Drug'][:-1]
plt.plot(corr)
plt.xticks(rotation=90)
plt.show()

# Machine Learning Model Building:

X = data_clean.drop('Drug', axis=1)
y = data_clean['Drug']
     

#test size 20% and train size 80%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

Random Forest:

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred, average='weighted')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='weighted')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='weighted')))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(rfc.score(X_test, y_test)*100)
plt.title(all_sample_title, size = 15)

#Feature Importance
imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": rfc.feature_importances_
})
fi = imp_df.sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(data=fi, x='Importance', y='Feature Name')
plt.title('Feature Importance Each Attributes', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

# AdaBoost:

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state=0)
ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred, average='weighted')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='weighted')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='weighted')))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(ada.score(X_test, y_test)*100)
plt.title(all_sample_title, size = 15)

#Feature Importance
imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": ada.feature_importances_
})
fi = imp_df.sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(data=fi, x='Importance', y='Feature Name')
plt.title('Feature Importance Each Attributes', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

