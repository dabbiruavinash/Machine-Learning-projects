import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)
pd.set_option('display.max_columns', None)
df = pd.read_csv('fraud_oracle.csv')
df

# Exploratory Data Analysis
sns.countplot(data=df, x="FraudFound_P", hue="BasePolicy")
#Liability BasePolicy are unlikely to do fraud vehicle insurance claim

sns.countplot(data=df, x="FraudFound_P", hue="NumberOfCars")

sns.histplot(data=df, x="Age", hue="FraudFound_P", multiple="stack")

sns.countplot(data=df, x="FraudFound_P", hue="VehiclePrice")

sns.countplot(data=df, x="FraudFound_P", hue="MaritalStatus")

sns.countplot(data=df, x="FraudFound_P", hue="VehicleCategory")
#People with sport car are less likely to do fraud nisurance claim

sns.countplot(data=df, x="FraudFound_P", hue="Sex")

sns.countplot(data=df, x="FraudFound_P", hue="Make")

sns.countplot(data=df, x="FraudFound_P", hue="AgeOfVehicle")

# Data Preprocessing
print(df.apply(lambda col: col.unique()))

df.select_dtypes(include='object').nunique()

df.drop(columns=['MonthClaimed','Month'], inplace=True)
df.head()

df['FraudFound_P'].unique()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['DayOfWeek']= label_encoder.fit_transform(df['DayOfWeek'])
df['DayOfWeek'].unique()

df['Make']= label_encoder.fit_transform(df['Make'])
df['Make'].unique()

df['AccidentArea']= label_encoder.fit_transform(df['AccidentArea'])
df['AccidentArea'].unique()

df['DayOfWeekClaimed']= label_encoder.fit_transform(df['DayOfWeekClaimed'])
df['DayOfWeekClaimed'].unique()

df['Sex']= label_encoder.fit_transform(df['Sex'])
df['Sex'].unique()

df['MaritalStatus']= label_encoder.fit_transform(df['MaritalStatus'])
df['MaritalStatus'].unique()

df['Fault']= label_encoder.fit_transform(df['Fault'])
df['Fault'].unique()

df['VehicleCategory']= label_encoder.fit_transform(df['VehicleCategory'])
df['VehicleCategory'].unique()

df['VehiclePrice']= label_encoder.fit_transform(df['VehiclePrice'])
df['VehiclePrice'].unique()

df['Days_Policy_Accident']= label_encoder.fit_transform(df['Days_Policy_Accident'])
df['Days_Policy_Accident'].unique()

df['Days_Policy_Claim']= label_encoder.fit_transform(df['Days_Policy_Claim'])
df['Days_Policy_Claim'].unique()

df['PastNumberOfClaims']= label_encoder.fit_transform(df['PastNumberOfClaims'])
df['PastNumberOfClaims'].unique()

df['AgeOfVehicle']= label_encoder.fit_transform(df['AgeOfVehicle'])
df['AgeOfVehicle'].unique()

df['AgeOfPolicyHolder']= label_encoder.fit_transform(df['AgeOfPolicyHolder'])
df['AgeOfPolicyHolder'].unique()

df['PoliceReportFiled']= label_encoder.fit_transform(df['PoliceReportFiled'])
df['PoliceReportFiled'].unique()

df['WitnessPresent']= label_encoder.fit_transform(df['WitnessPresent'])
df['WitnessPresent'].unique()

df['AgentType']= label_encoder.fit_transform(df['AgentType'])
df['AgentType'].unique()

df['NumberOfSuppliments']= label_encoder.fit_transform(df['NumberOfSuppliments'])
df['NumberOfSuppliments'].unique()

df['AddressChange_Claim']= label_encoder.fit_transform(df['AddressChange_Claim'])
df['AddressChange_Claim'].unique()

df['NumberOfCars']= label_encoder.fit_transform(df['NumberOfCars'])
df['NumberOfCars'].unique()

df['PolicyType']= label_encoder.fit_transform(df['PolicyType'])
df['PolicyType'].unique()

df['BasePolicy']= label_encoder.fit_transform(df['BasePolicy'])
df['BasePolicy'].unique()

df.head()

df.dtypes

#Check the target (FraudFound_P) if its balanced or not
sns.countplot(df['FraudFound_P'])
df['FraudFound_P'].value_counts()

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = df[(df['FraudFound_P']==0)] 
df_minority = df[(df['FraudFound_P']==1)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 14497, # to match majority class
                                 random_state=0)  # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])

sns.countplot(df_upsampled['FraudFound_P'])
df_upsampled['FraudFound_P'].value_counts()

#Remove the outlier using Z-Score
import scipy.stats as stats
z = np.abs(stats.zscore(df_upsampled))
data_clean = df_upsampled[(z<3).all(axis = 1)] 
data_clean.shape

# Machine Learning Model Building
X = data_clean.drop('FraudFound_P', axis=1)
y = data_clean['FraudFound_P']

#test size 20% and train size 80%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))

#Feature Importance
imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": dtree.feature_importances_
})

fi = imp_df.sort_values(by="Importance", ascending=False)
fi

fi2 = fi.head(10)
plt.figure(figsize=(10,8))
sns.barplot(data=fi2, x='Importance', y='Feature Name')
plt.title('Feature Importance Each Attributes (Decision Tree)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(dtree.score(X_test, y_test)*100)
plt.title(all_sample_title, size = 15)

from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = dtree.predict_proba(X_test)[:][:,1]
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

Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))

#Feature Importance
imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": rfc.feature_importances_
})

fi = imp_df.sort_values(by="Importance", ascending=False)
fi

fi2 = fi.head(10)
plt.figure(figsize=(10,8))
sns.barplot(data=fi2, x='Importance', y='Feature Name')
plt.title('Feature Importance Each Attributes (Random Forest)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

from sklearn.metrics import confusion_matrix
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

