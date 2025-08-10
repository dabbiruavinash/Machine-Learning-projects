import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, mean_absolute_error
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('customer churn.csv')
     
data.head(2)

data.shape

data.size

# EDA

data.info()

data.describe()
     
data.columns

data.duplicated().sum()

data.isna().sum()

# Data Cleaning

data.duplicated().sum()

data.isnull().sum()

for i in data.columns:
  if data[i].dtypes != object:
    plt.boxplot(data[i])
    plt.xlabel(i)
    plt.show()

#  Tasks

Extract the 5th column & store it in ‘customer_5’

customer_5 = data.iloc[:,4]
customer_5.sample(2)

Extract the 15th column & store it in ‘customer_15’

customer_15 = data.iloc[:,14]
customer_15.sample(2)

#To check Index of Columns
pp = list(data.columns)
for x,y in enumerate(pp):
  print(x,y)

Extract all the male senior citizens whose Payment Method is Electronic check & store the result in ‘senior_male_electronic’

senior_male_electronics = data[(data['gender'] == 'Male') & (data['SeniorCitizen'] == 1) &
                               (data['PaymentMethod'] == 'Electronic check')]

senior_male_electronics.sample(3)

Extract all those customers whose tenure is greater than 70 months or their Monthly charges is more than 100$ & store the result in ‘customer_total_tenure’

customer_total_tenure = data[(data['tenure']> 70) | (data['MonthlyCharges']>100)]
customer_total_tenure.sample(2)

Extract all the customers whose Contract is of two years, payment method is Mailed check & the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’

two_mail_yes = data[(data['Contract'] == 'Two year') & (data['PaymentMethod'] == 'Mailed check') & (data['Churn'] =='Yes')]
two_mail_yes.sample(2)

Extract 333 random records from the customer_churn dataframe& store the result in ‘customer_333’

customer_333 = data.sample(333)
customer_333.sample(5)

Get the count of different levels from the ‘Churn’ column

data.Churn.value_counts()

# Data Visualization:

Build a bar-plot for the ’InternetService’ column:

x = data.InternetService.value_counts().index
y = data.InternetService.value_counts()
     
plt.bar(x,y, color = ['blue', 'orange', 'red'])
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of Categories')
plt.show()
plt.tight_layout();

Build a histogram for the ‘tenure’ column:
Set the number of bins to be 30
Set the color of the bins to be ‘green’
Assign the title ‘Distribution of tenure’

plt.title('Distribution of tenure')
plt.hist(data.tenure, bins = 30, color = 'green')
plt.tight_layout();

Build a scatter-plot between ‘MonthlyCharges’ & ‘tenure’. Map ‘MonthlyCharges’ to the y-axis & ‘tenure’ to the ‘x-axis’:
Assign the points a color of ‘brown’
Set the x-axis label to ‘Tenure of customer’
Set the y-axis label to ‘Monthly Charges of customer’
Set the title to ‘Tenure vs Monthly Charges’

plt.title = 'Tenure vs Monthly Charges'
sns.scatterplot(y = 'MonthlyCharges', x = 'tenure', data = data, color = 'brown')
plt.xlabel = 'Tenure of customer'
plt.ylabel = 'Monthly Charges of customer'
plt.tight_layout()
plt.show();

data.boxplot('tenure',by=['Contract'])
plt.tight_layout()
     
Boxplot is an in-built function of plot

# We can play around with Boxplot

data.boxplot(column = 'tenure', by='Contract', figsize = (6,6));

Linear Regression:
Build a simple linear model where dependent variable is ‘MonthlyCharges’ and independent variable is ‘tenure’

Divide the dataset into train and test sets in 70:30 ratio.
Build the model on train set and predict the values on test set
After predicting the values, find the root mean square error
Find out the error in prediction & store the result in ‘error’
Find the root mean square error

Split Data

#Please Note to convert 1D Array into 2D use double Square Brackets

y = data[['MonthlyCharges']]
X = data[['tenure']]

y.shape
     
(7043, 1)

X.shape
     
(7043, 1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

# Model Building

LR = LinearRegression()     

LR.fit(X_train, y_train)
     
LinearRegression()

Predictions = LR.predict(X_test)
     
plt.scatter(Predictions, y_test);

Model Evaluation

LR.intercept_
     
LR.coef_

MAE = mean_absolute_error(y_test, Predictions)
MAE

Variance = mean_squared_error(y_test, Predictions)  #less MSE = Good Model | 0 MSE = Perfect Model
Variance

Standard_Deviation = np.sqrt(mean_squared_error(y_test, Predictions))
Standard_Deviation

r2_score(y_test, Predictions) #Goodness of Fit is preffered for Multiple linear regression

#Accuracy score is used for Logistic Regression

Useful Method for Logistic Regression with Muti Independent Variables

import statsmodels.api as sm
     
X = sm.OLS(y_train, X_train).fit()
X.summary()
     
Logistic Regression:
A. Build a simple logistic regression model where dependent variable is ‘Churn’ & independent variable is ‘MonthlyCharges’

Divide the dataset in 65:35 ratio
Build the model on train set and predict the values on test set
Build the confusion matrix and get the accuracy score

Data Split

y = data[['Churn']]
x = data[['MonthlyCharges']]
     
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.35, random_state=0)

Evaluating Model

classification_report(predictions, y_test) #Gives report of Accuracy, Precision, Recall, F1-Score

confusion_matrix(predictions, y_test)

accuracy_score(predictions, y_test)

Build a multiple logistic regression model where dependent variable is ‘Churn’ & independent variables are ‘tenure’ & ‘MonthlyCharges’

Divide the dataset in 80:20 ratio
Build the model on train set and predict the values on test set
Build the confusion matrix and get the accuracy score
Data Split

y = data[['Churn']]
x = data[['MonthlyCharges', 'tenure']]
     
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=0)

Building Model

LM = LogisticRegression()
     
LM.fit(x_train, y_train)
     
LogisticRegression()

predictions = LM.predict(x_test)

Model Evaluation

accuracy_score(predictions,y_test)
     
0.7735982966643009

confusion_matrix(predictions,y_test)
     
array([[934, 212],
       [107, 156]])

classification_report(predictions,y_test)