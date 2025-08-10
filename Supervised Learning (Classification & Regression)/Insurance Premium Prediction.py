import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stat
import math
import warnings
warnings.filterwarnings('ignore')

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import *

data =pd.read_csv('new_insurance_data.csv')

# EDA:

data.head()

data.tail()

data.shape

data.columns

data.info()

data.isnull().sum()

data.isnull().mean() * 100

data.corr()

# Visualization:

#visualizing null values
plt.figure(figsize = (5,5))
sns.heatmap(data.isnull(), cmap= 'viridis');

sns.pairplot(data);

#distribution of target 
sns.distplot(data.charges);

sns.heatmap(data.corr(), cmap = 'viridis');

# Data Preparation:

data.duplicated().sum()

# Missing Values:

#for float and int values we prefer using mean
#for object type values we prefer using modefor col_name in col:

cols = data.columns

for i in cols:    #Using for loop
    if data[i].dtypes == object:
        data[i] = data[i].fillna(data[i].mode()[0])
    else:
        data[i] = data[i].fillna(data[i].mean())

data.isna().sum()  #after mean/median imputation

# Outliers:

for i in cols:
    if data[i].dtypes == object:
      pass
    else:
      plt.boxplot(data[i])
      plt.xlabel(i)
      plt.ylabel('Count')
      plt.show()

# Treating Outliers:

#For Bmi

Q1 = data.bmi.quantile(0.25)
Q3 = data.bmi.quantile(0.75)
IQR = Q3-Q1
data = data[(data.bmi >= Q1 - 1.5*IQR) & (data.bmi <= Q3 + 1.5*IQR)]

#For past_consultations

Q1 = data.past_consultations.quantile(0.25)
Q3 = data.past_consultations.quantile(0.75)
IQR = Q3-Q1
data = data[(data.past_consultations >= Q1 - 1.5*IQR) & (data.past_consultations <= Q3 + 1.5*IQR)]

#For hospital_expenditure

Q1=data.Hospital_expenditure.quantile(0.25)
Q3=data.Hospital_expenditure.quantile(0.75)
IQR=Q3-Q1
data=data[(data.Hospital_expenditure>=Q1 - 1.5*IQR) & (data.Hospital_expenditure<=Q3 + 1.5*IQR)]

#For Anual_Salary

Q1=data.Anual_Salary.quantile(0.25)
Q3=data.Anual_Salary.quantile(0.75)
IQR=Q3-Q1
data=data[(data.Anual_Salary>=Q1 - 1.5*IQR) & (data.Anual_Salary<=Q3 + 1.5*IQR)]

#cheking outlier again 

for i in cols:
    if data[i].dtypes==object:
        pass
    else:
        plt.boxplot(data[i])
        plt.xlabel(i)
        plt.ylabel('count')
        plt.show()

# Multi Collinearity:

C = data.corr()
sns.heatmap(data = C);

from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list=[]
for col in data.columns:
    if((data[col].dtypes!=object)):
        col_list.append(col)
        
col_list

X = data[col_list]
X.columns

#cheking VIF
vif_data=pd.DataFrame()
vif_data['feature']=X.columns
vif_data["VIF"]=[variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
print(vif_data)

#dropping columns with high VIF
data=data.drop(['num_of_steps','NUmber_of_past_hospitalizations','age','bmi'],axis=1)
#Comparing to above values of Multi collinearity we have now a reduced score of Collinearity
col_list=[]
for col in data.columns:
    if((data[col].dtypes!=object)&(col!='charges')):
        col_list.append(col)
        
X=data[col_list]

vif_data=pd.DataFrame()
vif_data['feature']=X.columns
vif_data["VIF"]=[variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
print(vif_data)

# Data Pre Processing:

x=data.loc[:,['children','Claim_Amount','past_consultations','Hospital_expenditure','Anual_Salary']]
y=data.iloc[:,-1]

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

#train_size  - 80% of the data for train Data set
#random_state = could be 0, 1, 85 etc but 42 pattern is standard

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Model Building:

# Linear Regression
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x_train, y_train) 

#To get predictions
y_pred=linear_reg.predict(x_test)

from sklearn.metrics import *

linear_reg_mse = mean_squared_error(y_test, y_pred)
linear_reg_rmse = mean_squared_error(y_test, y_pred, squared=False)
linear_reg_r2_score = r2_score(y_test, y_pred)

# Evaluation Metrics
print(f"The Mean Squared Error using Linear Regression : ", linear_reg_mse)
print(f"The Root Mean Squared Error using Linear Regression : ", linear_reg_rmse)
print(f"The r2_score using Linear Regression : ", linear_reg_r2_score)

# Decision Tree Regressor:

from sklearn.tree import DecisionTreeRegressor
decision_tree= LinearRegression()
decision_tree.fit(x_train, y_train)

#To get predictions
y_pred1 = decision_tree.predict(x_test)

# Evaluation Metrics
decision_tree_mse = mean_squared_error(y_test, y_pred)
decision_tree_rmse = mean_squared_error(y_test, y_pred1, squared=False)
decision_tree_r2_score = r2_score(y_test, y_pred1)

print(f"The Mean Squared Error using Decision Tree Regressor : ",decision_tree_mse)
print(f"The Root Mean Squared Error using Decision Tree Regressor : ", decision_tree_rmse)
print(f"The r2_score using Decision Tree Regressor : ", decision_tree_r2_score)

# Random Forest Regressor:

from sklearn.ensemble import RandomForestRegressor

random_forest= RandomForestRegressor()
random_forest.fit(x_train, y_train)

#To get predictions
y_pred2 = decision_tree.predict(x_test)
# Evaluation Metrics
random_forest_mse = mean_squared_error(y_test, y_pred2)
random_forest_rmse = mean_squared_error(y_test, y_pred2, squared=False)
random_forest_r2_score = r2_score(y_test, y_pred2)

print(f"The Mean Squared Error using Random Forest Regressor : ", random_forest_mse)
print(f"The Root Mean Squared Error using Random Forest Regressor : ", random_forest_rmse)
print(f"The r2_score Error using Random Forest Regressor : ", random_forest_r2_score)

# Gradient Boosting:

gradient_boosting_reg = GradientBoostingRegressor()

gradient_boosting_reg.fit(x_train, y_train)

#To get predictions
y_pred3 = gradient_boosting_reg.predict(x_test)
# Evaluation Metrics
gradient_boosting_mse = mean_squared_error(y_test, y_pred3)
gradient_boosting_rmse = mean_squared_error(y_test, y_pred3, squared=False)
gradient_boosting_r2_score = r2_score(y_test, y_pred3)

print(f"The Mean Squared Error using Gradient Boosting Regressor : ", gradient_boosting_mse)
print(f"The Root Mean Squared Error using Gradient Boosting Regressor : ", gradient_boosting_rmse)
print(f"The r2_sccore using Gradient Boosting Regressor : ",gradient_boosting_r2_score)

# KNN:

knn = KNeighborsRegressor(n_neighbors=10)

knn.fit(x_train, y_train)

#To get predictions
y_pred4 = knn.predict(x_test)
# Evaluation Metrics
knn_mse = mean_squared_error(y_test, y_pred4)
knn_rmse = mean_squared_error(y_test, y_pred4, squared=False)
knn_r2_score = r2_score(y_test, y_pred4)

print(f"The mean squared error using KNN is ",knn_mse)
print(f"The root mean squared error using KNN is ",knn_rmse)
print(f"The r2_score using KNN is ",knn_r2_score)

# XGBoost:

xgb = xgb.XGBRegressor()
xgb.fit(x_train, y_train)

#To get predictions
y_pred5 = xgb.predict(x_test)
# Evaluation Metrics
xgb_reg_mse = mean_squared_error(y_test, y_pred5)
xgb_reg_rmse = mean_squared_error(y_test, y_pred5, squared=False)
xgb_reg_r2_score = r2_score(y_test, y_pred5)

print(f"The mean square error using XGBoost is ",xgb_reg_mse)
print(f"The root mean_squared error using XGBoost is ",xgb_reg_rmse)
print(f"The r2 score using XGBoost is ", xgb_reg_r2_score)

# To Get Best Performing Model:

models = pd.DataFrame({
    'Model' : ['Linear Regression', 'Decision Tree', 'Random Forest',
               'Gradient Boosting', 'KNN', 'XGBoost'],
    'RMSE' : [linear_reg_rmse, decision_tree_rmse, random_forest_rmse,
            gradient_boosting_rmse, knn_rmse, xgb_reg_rmse],
    'r2_score' : [linear_reg_r2_score, decision_tree_r2_score, random_forest_r2_score, 
    gradient_boosting_r2_score, knn_r2_score, xgb_reg_r2_score]
})

models.sort_values(by='RMSE', ascending=True)

