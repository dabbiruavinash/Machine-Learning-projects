import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(color_codes=True)
     
Dataset : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head()

# Exploratory Data Analysis:

df.isnull().sum()
     
df.dtypes

df['bmi'].fillna(int(df['bmi'].mean()), inplace=True)
df.isnull().sum()

df2 = df.drop('id', axis=1)
df2.head()

#Change value in ever_married column
df2['ever_married'] = df2['ever_married'].replace(['No'],'0')
df2['ever_married'] = df2['ever_married'].replace(['Yes'],'1')
#Change value in work_type column
df2['work_type'] = df2['work_type'].replace(['children'],'0')
df2['work_type'] = df2['work_type'].replace(['Govt_job'],'1')
df2['work_type'] = df2['work_type'].replace(['Never_worked'],'2')
df2['work_type'] = df2['work_type'].replace(['Private'],'3')
df2['work_type'] = df2['work_type'].replace(['Self-employed'],'4')
#Change value in Residence_type column
df2['Residence_type'] = df2['Residence_type'].replace(['Rural'],'0')
df2['Residence_type'] = df2['Residence_type'].replace(['Urban'],'1')
#Change value in smoking_status column
df2['smoking_status'] = df2['smoking_status'].replace(['never smoked'],'0')
df2['smoking_status'] = df2['smoking_status'].replace(['formerly smoked'],'1')
df2['smoking_status'] = df2['smoking_status'].replace(['smokes'],'2')
df2['smoking_status'] = df2['smoking_status'].replace(['Unknown'],'3')
#Change value in gender column
df2['gender'] = df2['gender'].replace(['Female'],'0')
df2['gender'] = df2['gender'].replace(['Male'],'1')
df2['gender'] = df2['gender'].replace(['Other'],'2')
df2.head()

#convert object data types column to integer
df2['gender'] = pd.to_numeric(df2['gender'])
df2['ever_married'] = pd.to_numeric(df2['ever_married'])
df2['work_type'] = pd.to_numeric(df2['work_type'])
df2['Residence_type'] = pd.to_numeric(df2['Residence_type'])
df2['smoking_status'] = pd.to_numeric(df2['smoking_status'])
df2.dtypes
     
#Counting 1 and 0 Value in stroke column
sns.countplot(df2['stroke'])

# Oversampling data:

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = df2[(df2['stroke']==0)] 
df_minority = df2[(df2['stroke']==1)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 4861, # to match majority class
                                 random_state=42)  # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])
     
sns.countplot(df_upsampled['stroke'])

sns.heatmap(df_upsampled.corr(), fmt='.2g')

# Build Machine Learning Model

X = df_upsampled.drop('stroke', axis=1)
y = df_upsampled['stroke']
     
#test size 20% and train size 80%
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=7)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

# XGBoost

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

# Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

# Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

#importing classification report and confussion maatrix from sklearn
from sklearn.metrics import classification_report, confusion_matrix

# XGBoost

y_pred = xgb.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))

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
     
from matplotlib import pyplot
model = XGBClassifier()
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()

model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train,y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

from xgboost import plot_importance
from matplotlib import pyplot
plot_importance(model)
plt.figure(figsize=(30,45))
pyplot.show()

import xgboost as xgb
plt.figure(figsize=(20,20))
xgb.plot_tree(model, ax=plt.gca());

# Logistic Regression

y_pred = lr.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(lr.score(X_test, y_test)*100)
plt.title(all_sample_title, size = 15)

from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = lr.predict_proba(X_test)[:][:,1]
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

y_pred_prob = lr.predict_proba(X_test)[0:10]
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of dont have stroke', 'Prob of have stroke'])
y_pred_prob_df
     
y_pred1 = lr.predict_proba(X_test)[:, 1]
# adjust the font size 
plt.rcParams['font.size'] = 12
# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of stroke')
# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of stroke')
plt.ylabel('Frequency')

# Predicting Stroke by Inputing number (Value)

Xnew = [[1, 50, 0, 1, 0, 3, 1, 140, 25, 2]]
y_pred_prob2 = lr.predict_proba(Xnew)
y_pred_prob_df2 = pd.DataFrame(data=y_pred_prob2, columns=['Prob of dont have stroke', 'Prob of have stroke'])
y_pred_prob_df2
     
# Random Forest

y_pred = rfc.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(rfc.score(X_test, y_test)*100)
plt.title(all_sample_title, size = 15)

from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = rfc.predict_proba(X_test)[:][:,1]
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
     

#Feature Importance
imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": rfc.feature_importances_
})

imp_df.sort_values(by="Importance", ascending=False)

y_pred_prob = rfc.predict_proba(X_test)[0:10]
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of dont have stroke', 'Prob of have stroke'])
y_pred_prob_df
     
y_pred2 = rfc.predict_proba(X_test)[:, 1]
# adjust the font size 
plt.rcParams['font.size'] = 12
# plot histogram with 10 bins
plt.hist(y_pred2, bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of stroke')
# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of stroke')
plt.ylabel('Frequency')

# Predicting Stroke by Inputing number (Value)

Xnew2 = [[1, 70, 1, 1, 0, 3, 1, 100, 40, 2]]
y_pred_prob3 = rfc.predict_proba(Xnew2)
y_pred_prob_df3 = pd.DataFrame(data=y_pred_prob3, columns=['Prob of dont have stroke', 'Prob of have stroke'])
y_pred_prob_df3

# Taking input from the user
gender = input("Your Gender (0 = Female, 1 = Male) : ")
age = input("Your Age : ")
hypertension = input("Do you have hypertension ? (0 = No, 1 = Yes) : ")
heart = input("Do you have heart disease ? (0 = No, 1 = Yes) :")
marry = input("Did you ever married ? (0 = No, 1 = Yes) :")
work = input("Your Worktype ? (0 = children, 1 = Government job, 2 = Never worked, 3 = Private, 4 = Self Employed) : ")
residence = input("Your Residence type ? (0 = Rural, 1 = Urban) : ")
avg = input("Average Glucose Level : ")
bmi = input("Your BMI : ")
smoke = input("Your Smoking status ? (0 = never smoked, 1 = formerly smoked, 2 = smokes, 3 = unknown) : ")

Xnew3 = [[gender, age, hypertension, heart, marry, work, residence, avg, bmi, smoke]]


y_pred_prob4 = rfc.predict_proba(Xnew3)
y_pred_prob_df4 = pd.DataFrame(data=y_pred_prob4, columns=['Prob of dont have stroke', 'Prob of have stroke'])
y_pred_prob_df4

