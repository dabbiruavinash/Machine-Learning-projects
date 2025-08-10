import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)
     
Dataset : https://www.kaggle.com/datasets/shantanuss/banknote-authentication-uci

df = pd.read_csv('BankNote_Authentication.csv')
df

sns.boxplot(x=df["variance"])

sns.boxplot(x=df["skewness"])

sns.boxplot(x=df["curtosis"])

sns.boxplot(x=df["entropy"])

import scipy.stats as stats
z = np.abs(stats.zscore(df))
data_clean = df[(z<2).all(axis=1)]
data_clean.shape
     
#Counting 1 and 0 Value in class column
sns.countplot(data_clean['class'])
data_clean['class'].value_counts()

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = data_clean[(data_clean['class']==0)] 
df_minority = data_clean[(data_clean['class']==1)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 727, # to match majority class
                                 random_state=42)  # reproducible results
# Combine majority class with upsampled minority class
data_clean2 = pd.concat([df_minority_upsampled, df_majority])
     

#Counting 1 and 0 Value in class column
sns.countplot(data_clean2['class'])
data_clean2['class'].value_counts()

sns.heatmap(data_clean2.corr(), annot=True)

X = data_clean2.drop('class', axis=1)
y = data_clean2['class']
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from sklearn.metrics import accuracy_score
     

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
     
y_pred = dtree.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

# Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
     
y_pred = rfc.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

# XGBoost:

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

# Logistic Regression:

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
     
y_pred = lr.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

# Visualize XGBoost

from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(xgb.score(X_test, y_test)*100)
plt.title(all_sample_title, size = 15)

from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = xgb.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_test.index

fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', size = 15)
plt.legend()

from xgboost import plot_importance
from matplotlib import pyplot
plot_importance(xgb)
plt.figure(figsize=(30,45))
pyplot.show()

model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train,y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

import xgboost as xgb
plt.figure(figsize=(20,20))
xgb.plot_tree(model, ax=plt.gca());

