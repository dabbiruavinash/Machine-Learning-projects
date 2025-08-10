import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)
pd.set_option('display.max_columns', None)
df = pd.read_csv('Weather_Data.csv')
df.head()

df.info()

df['Date'] = df['Date'].astype('datetime64[ns]')
df.head()

# Exploartory Data Analysis:

x = df['Date']
y = df['Rainfall']

# plot
plt.plot(x,y)
# beautify the x-labels
plt.gcf().autofmt_xdate()

plt.show()

sns.displot(x='WindGustDir', hue='RainTomorrow', data=df, multiple='stack')
plt.xticks (rotation='vertical')

sns.boxplot(data=df, x="RainTomorrow", y="Rainfall")

sns.boxplot(data=df, x="RainTomorrow", y="Sunshine")

sns.boxplot(data=df, x="RainTomorrow", y="WindGustSpeed")

sns.displot(x='RainToday', hue='RainTomorrow', data=df, multiple='fill', stat="density")

sns.scatterplot(data=df, x="MinTemp", y="MaxTemp", hue="RainTomorrow")

sns.scatterplot(data=df, x="Rainfall", y="Evaporation", hue="RainTomorrow")

# Data Preprocessing:

df.dtypes

df['WindGustDir'].unique()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['WindGustDir']= label_encoder.fit_transform(df['WindGustDir'])
df['WindGustDir'].unique()

df['WindDir9am'].unique()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['WindDir9am']= label_encoder.fit_transform(df['WindDir9am'])
df['WindDir9am'].unique()

df['WindDir3pm'].unique()

df['RainToday'].unique()

df['RainTomorrow'].unique()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['RainTomorrow']= label_encoder.fit_transform(df['RainTomorrow'])
df['RainTomorrow'].unique()

df.dtypes

df.head()

# Check The Class Value if its Balanced or Not:

#Counting 1 and 0 Value in Response column
sns.countplot(df['RainTomorrow'])
df['RainTomorrow'].value_counts()

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = df[(df['RainTomorrow']==0)] 
df_minority = df[(df['RainTomorrow']==1)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 2422, # to match majority class
                                 random_state=0)  # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])

sns.countplot(df_upsampled['RainTomorrow'])
df_upsampled['RainTomorrow'].value_counts()

# Remove the Outlier using Z-Score:

df_upsampled.drop(columns='Date', inplace=True)
df_upsampled.head()

#Remove Outlier using Z-Score Method
import scipy.stats as stats
z = np.abs(stats.zscore(df_upsampled))
data_clean = df_upsampled[(z<3).all(axis = 1)] 
data_clean.shape

# Check the Correlation:

sns.heatmap(data_clean.corr(), fmt='.2g')

# Training and Test Data:

X = data_clean.drop('RainTomorrow', axis=1)
y = data_clean['RainTomorrow']

#test size 20% and train size 80%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

# Logistic Regression:

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
print('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
print('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))
print('Log Loss : ',(log_loss(y_test, y_pred)))

# K - Nearest Neighbor:

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
print('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
print('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))
print('Log Loss : ',(log_loss(y_test, y_pred)))

# Support Vector Machine:

from sklearn import svm
support = svm.SVC()
support.fit(X_train, y_train)

y_pred = support.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
print('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
print('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))
print('Log Loss : ',(log_loss(y_test, y_pred)))

# Decision Tree:

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
print('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
print('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))
print('Log Loss : ',(log_loss(y_test, y_pred)))

# Confusion Matrix for All of Algorithms:

lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Logistic Regression: {0}'.format(lr.score(X_test, y_test))
plt.title(all_sample_title, size = 15)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for K Nearest Neighbors : {0}'.format(neigh.score(X_test, y_test))
plt.title(all_sample_title, size = 15)

support = svm.SVC()
support.fit(X_train, y_train)

y_pred = support.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Support Vector Machine : {0}'.format(support.score(X_test, y_test))
plt.title(all_sample_title, size = 15)

dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Decision Tree : {0}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 15)

# Feature Importance for Decision Tree
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
plt.title('Top 10 Feature Importance Each Attributes (Decision Tree)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

