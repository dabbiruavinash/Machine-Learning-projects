import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from sklearn.model_selection import train_test_split
%matplotlib inline
     

diabetes_df = pd.read_csv('diabetes.csv')
diabetes_df

# Exploratory Data Analysis

#Column in Dataset
diabetes_df.columns

#Show data types and null value each column
diabetes_df.info()

#Menampilkan 10 baris pertama
diabetes_df.isnull().head(10)

#Checking if there is null value
diabetes_df.isnull().sum()

#Checking if there is zero value

#replace 0 value with NaN
diabetes_df_copy = diabetes_df.copy(deep = True) #deep = True -> Buat salinan indeks dan data dalam dataframe
diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Showing the Count of NANs
print(diabetes_df_copy.isnull().sum())

#Fill null value with median
diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].median(), inplace = True)
diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].median(), inplace = True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace = True)
diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace = True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace = True)
#inplace = True -> untuk menyimpan hasil modifikasi pada dataframe diabetes_df_copy

#Counting 1 and 0 Value in Outcome column
sns.countplot(diabetes_df_copy['Outcome']) #membuat bar plot perbandingan jumlah value
print(diabetes_df_copy.Outcome.value_counts()) #menampilkan jumlah value 0 dan 1

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = diabetes_df_copy[(diabetes_df_copy['Outcome']==0)] # semua data yang value outcome nya = 0
df_minority = diabetes_df_copy[(diabetes_df_copy['Outcome']==1)] # semua data yang value outcome nya = 1
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 n_samples= 500, # to match majority class, menyamakan jumlah value 1 dengan 0
                                 random_state=0)  # reproducible results, random state 0 is better than 42
                                                  #Random state = Mengontrol pengacakan yang diterapkan ke data agar hasil yang didapatkan tetap sama
# Combine majority class with upsampled minority class
diabetes_df_copy2 = pd.concat([df_minority_upsampled, df_majority]) #menggabungkan Outcome 1 (minority) yang sudah di upsample dengan Outcome 0 (majority)
     
#Counting 1 and 0 Value in Outcome column
sns.countplot(diabetes_df_copy2['Outcome'])
print(diabetes_df_copy2.Outcome.value_counts())

# Checking Outliers using Box Plot

sns.boxplot(x=diabetes_df_copy2["Pregnancies"])
     
sns.boxplot(x=diabetes_df_copy2["Glucose"])

sns.boxplot(x=diabetes_df_copy2["BloodPressure"])

sns.boxplot(x=diabetes_df_copy2["SkinThickness"])

sns.boxplot(x=diabetes_df_copy2["Insulin"])

sns.boxplot(x=diabetes_df_copy2["BMI"])

sns.boxplot(x=diabetes_df_copy2["DiabetesPedigreeFunction"])

sns.boxplot(x=diabetes_df_copy2["Age"])

# Check Outlier From Scratch using Z-Score

out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(diabetes_df_copy2["Pregnancies"])


out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(diabetes_df_copy2["BloodPressure"])

out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(diabetes_df_copy2["SkinThickness"])

out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(diabetes_df_copy2["Insulin"])

out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(diabetes_df_copy2["BMI"])

out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(diabetes_df_copy2["Age"])

out=[]
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    for i in df: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(diabetes_df_copy2["DiabetesPedigreeFunction"])

import scipy.stats as stats
z = np.abs(stats.zscore(diabetes_df_copy2))
data_clean = diabetes_df_copy2[(z<3).all(axis = 1)] #print all of rows that have z<3 (z score below 3)
data_clean.shape

#Cleaned Outliers data using Z Scores
data_clean

diabetes_df_copy2

#Print rows dalam dataframe diabetes_df_copy2 yang not isin(tidak didalam) dataframe data_clean
#lambang (~) menandakan NOT
diabetes_df_copy2[~diabetes_df_copy2.index.isin(data_clean.index)]

# Data Correlation

sns.heatmap(data_clean.corr(), annot=True)

#.corr() = correlation matrix

X = data_clean.drop('Outcome', axis=1) 
y = data_clean['Outcome'] 


#test size 10% and train size 90%
#Random state 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=0) 

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(random_state=0) 
dtree.fit(X_train, y_train)
     
DecisionTreeClassifier(random_state=0)

from sklearn.metrics import confusion_matrix
     
y_pred = dtree.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))

cm = confusion_matrix(y_test, y_pred) 
plt.figure(figsize=(5,5)) 
#settingan heatmap, 

sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues') 
plt.ylabel('Actual label') #Mendefinisikan sumbu y untuk Actual label
plt.xlabel('Predicted label') #Mendefinisikan sumbu x untuk predicted label
all_sample_title = 'Accuracy Score: {0}'.format(dtree.score(X_test, y_test)) 
plt.title(all_sample_title, size = 15) 

# XGBoost

from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
     
XGBClassifier()

y_pred = xgb_model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))
     
F-1 Score :  0.7959183673469388
Precision Score :  0.7959183673469388
Recall Score :  0.7959183673469388

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(xgb_model.score(X_test, y_test)) 
plt.title(all_sample_title, size = 15)

# Improving the Classification Accuracy using Recursive Feature Elimination with Cross-Validation

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)
     
DecisionTreeClassifier(random_state=0)

y_pred = dtree.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))
     

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 15)

# XGBoost:

from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
     
XGBClassifier()

y_pred = xgb_model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))
     
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(xgb_model.score(X_test, y_test))
plt.title(all_sample_title, size = 15)
     

# Comparative Analysis for Diabetic Prediction Based on Machine Learning Techniques:

#test size 30% and train size 70%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

