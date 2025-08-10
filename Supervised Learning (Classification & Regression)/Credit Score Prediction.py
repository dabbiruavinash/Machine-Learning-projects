import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r'credit score.csv')
df.head(3)

df.shape

df.info()
# Column SSN--> Social Security Number

df.describe().T

df.isna().sum()

sns.boxplot(x=df['Credit_Score'], y=df['Monthly_Inhand_Salary'],data=df )

df1=df.copy()
df1.head()

df1.info()

df1['Annual_Income']= df1['Annual_Income'].str.replace('_','').astype(float)
df1['Annual_Income'].dtype

df1['Monthly_Inhand_Salary']= df1['Monthly_Inhand_Salary'].replace(np.nan, 0)
df1['Monthly_Inhand_Salary'].isna().sum()

df1['Num_Bank_Accounts'].isna().sum()

df1['Num_Credit_Card'].unique()

df1['Interest_Rate'].dtype

df1['Num_of_Loan'].unique()

df1['Num_of_Loan']=df1['Num_of_Loan'].str.replace('_','').astype(int)
df1['Num_of_Loan'].dtype

df1['Delay_from_due_date'].unique()

df1['Delay_from_due_date']=df1['Delay_from_due_date'].replace('-','')
df1['Delay_from_due_date'].dtype

df1['Num_of_Delayed_Payment'].unique()

df1['Num_of_Delayed_Payment']=df1['Num_of_Delayed_Payment'].str.replace('_','')
df1['Num_of_Delayed_Payment']= df1['Num_of_Delayed_Payment'].replace('-','')
df1['Num_of_Delayed_Payment'].unique()

df1['Num_of_Delayed_Payment']=df1['Num_of_Delayed_Payment'].replace(np.nan,0)
df1['Num_of_Delayed_Payment'].isna().sum()

df1['Num_of_Delayed_Payment']=df1['Num_of_Delayed_Payment'].str.replace('-','')
df1['Num_of_Delayed_Payment'].unique()

df1['Credit_Mix'].unique()

df1['Credit_Mix']= df1['Credit_Mix'].replace('_','None')
df1['Credit_Mix'].unique()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df1['Credit_Mix'] = le.fit_transform(df1['Credit_Mix'])
df1['Credit_Mix'].unique()

df1['Outstanding_Debt'].unique()

df1['Outstanding_Debt']=df1['Outstanding_Debt'].str.replace('_','').astype(float)
df1['Outstanding_Debt'].dtype

df1['Outstanding_Debt'].unique()

df1['Credit_History_Age']

#df['Credit_History_Age']=df['Credit_History_Age'].str.extract('(^\d*)')
df1['Credit_History_Age']=df1['Credit_History_Age'].str.findall(r'(\d+(?:\.\d+)?)')
df1['Credit_History_Age']=df1['Credit_History_Age'].replace(np.nan,  0)
df1['Credit_History_Age'].isna().sum()

df1['Credit_History_Age']

#pd.DataFrame(df1['Credit_History_Age'].to_list(), columns=['Year','Month'])
df1['Monthly_Balance'].unique()

df1['Monthly_Balance']= df1['Monthly_Balance'].str.replace('_','')
df1['Monthly_Balance']=df1['Monthly_Balance'].astype('float')
df1['Monthly_Balance'].dtype

df1['Monthly_Balance']=df1['Monthly_Balance'].replace(np.nan,0)
df1['Credit_Score'].unique()

df1['Credit_Score'].isna().sum()

df1['Credit_Score']=le.fit_transform(df['Credit_Score'])
df1['Credit_Score'].unique()

from sklearn.model_selection import train_test_split
x=df1[['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
       'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
       'Delay_from_due_date', 'Num_of_Delayed_Payment',
       'Credit_Mix', 'Outstanding_Debt',
        'Monthly_Balance']]

x['Num_of_Delayed_Payment']=x['Num_of_Delayed_Payment'].replace(np.nan,0)
x.isna().sum()

y=df1[['Credit_Score']]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

y_pred= lr.predict(x_test)
y_pred

from sklearn.metrics import classification_report, accuracy_score
cr= classification_report(y_pred, y_test)
print(cr)

y_pred= dt.predict(x_test)
cr=classification_report(y_pred,y_test)
print(cr)

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(x_train, y_train)

y_pred=rf.predict(x_test)
cr=classification_report(y_pred, y_test)
print(cr)

accuracy_score(y_pred,y_test)

df1['Credit_Score'].value_counts()     # n_components=3-1=2

# LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
x_train= lda.fit_transform(x_train,y_train)
x_test=lda.transform(x_test)
x_train.shape

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

y_pred= dt.predict(x_test)
cr=classification_report(y_pred, y_test)
print(cr)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train, y_train)

y_pred= rf.predict(x_test)
cr= classification_report(y_pred, y_test)
print(cr)
