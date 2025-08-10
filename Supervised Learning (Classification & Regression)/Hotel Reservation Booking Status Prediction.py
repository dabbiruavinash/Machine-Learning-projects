import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True)
df = pd.read_csv('Hotel Reservations.csv')
df

# Exploratory Data Analysis:

sns.countplot(data=df, x="booking_status", hue="market_segment_type")

sns.countplot(data=df, x="booking_status", hue="room_type_reserved")

sns.countplot(data=df, x="booking_status", hue="type_of_meal_plan")

sns.countplot(data=df, x="booking_status", hue="arrival_year")

sns.countplot(data=df, x="booking_status", hue="arrival_month")

sns.barplot(data=df, x="booking_status", y="avg_price_per_room")
# Customer with higher price per room more likely to canceled the order

sns.barplot(data=df, x="booking_status", y="lead_time")
#Customer with higher lead time are more likely to cancel the order

# Data Preprocessing Part:

df.dtypes

df.drop(columns='Booking_ID',inplace=True)
df.shape

df['type_of_meal_plan'].unique()

df['room_type_reserved'].unique()

df['market_segment_type'].unique()

df['booking_status'].unique()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['type_of_meal_plan']= label_encoder.fit_transform(df['type_of_meal_plan'])
df['type_of_meal_plan'].unique()

df['room_type_reserved']= label_encoder.fit_transform(df['room_type_reserved'])
df['room_type_reserved'].unique()

df['market_segment_type']= label_encoder.fit_transform(df['market_segment_type'])
df['market_segment_type'].unique()

df['booking_status']= label_encoder.fit_transform(df['booking_status'])
df['booking_status'].unique()

df.dtypes

df.head()

# Check if the Class data is balanced or not

sns.countplot(df['booking_status'])
df['booking_status'].value_counts()

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = df[(df['booking_status']==1)] 
df_minority = df[(df['booking_status']==0)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 24390, # to match majority class
                                 random_state=0)  # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])

sns.countplot(df_upsampled['booking_status'])
df_upsampled['booking_status'].value_counts()

# Check the Outlier:

sns.boxplot(x=df_upsampled["lead_time"])

sns.boxplot(x=df_upsampled["avg_price_per_room"])

#Remove the outlier using Z-Score
import scipy.stats as stats
z = np.abs(stats.zscore(df_upsampled))
data_clean = df_upsampled[(z<3).all(axis = 1)] 
data_clean.shape

sns.heatmap(data_clean.corr(), fmt='.2g')

#Remove unnecesary attribute
cols_to_drop = [
    'required_car_parking_space',
    'repeated_guest',
    'no_of_previous_cancellations',
    'no_of_previous_bookings_not_canceled'
]
data_clean.drop(columns=cols_to_drop,inplace=True)
data_clean.head()

# Machine Learning Model Building
X = data_clean.drop('booking_status', axis=1)
y = data_clean['booking_status']

#test size 20% and train size 80%
from sklearn.model_selection import train_test_split
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

# Random Forest Classifier
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

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state=0)
ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print('F-1 Score : ',(f1_score(y_test, y_pred)))
print('Precision Score : ',(precision_score(y_test, y_pred)))
print('Recall Score : ',(recall_score(y_test, y_pred)))

#Feature Importance
imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": ada.feature_importances_
})

fi = imp_df.sort_values(by="Importance", ascending=False)
fi

fi2 = fi.head(10)
plt.figure(figsize=(10,8))
sns.barplot(data=fi2, x='Importance', y='Feature Name')
plt.title('Feature Importance Each Attributes (AdaBoost)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

