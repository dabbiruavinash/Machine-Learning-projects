import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('/content/drive/MyDrive/ML Project/My project/IPL/Data/matches.csv')

# EDA

df.shape

df.head()

df.info()

df.describe()

df.isna().sum()

df = df.drop(['umpire3'], axis=1)
df.head()

df.dropna()

df.shape

df['team1'].unique()

df['team1']=df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
df['team2']=df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
df['winner']=df['winner'].str.replace('Delhi Daredevils','Delhi Capitals')
     

df['team1']=df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
df['team2']=df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
df['winner']=df['winner'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

# Data Visulization:

plt.figure(figsize = (20,6))
sns.countplot(y='winner', data=df, order=df['winner'].value_counts().index)
plt.xlabel('Wins')
plt.ylabel('Team')
plt.title('Number of  IPL  matches won by each team')

plt.figure(figsize = (20,20))
sns.countplot(y='venue', data=df, order=df['venue'].value_counts().index)
plt.xlabel('No of matches',fontsize=12)
plt.ylabel('Venue',fontsize=12)
plt.title('Total Number of matches played in different stadium')

plt.figure(figsize=(8,5))
sns.countplot(x='toss_decision', data=df)
plt.xlabel('Toss Decesion', fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.title('Toss Decesion')

# Data Modification:

x = ["city", "toss_decision", "result", "dl_applied"]
for i in x:
  print("------------")
  print(df[i].unique())
  print(df[i].value_counts())
     

df.drop(["id", "Season","city","date", "player_of_match", 'umpire1', "venue", "umpire2"], axis=1, inplace=True)
df.head()

# Spliting Dataset
x = df.drop(['winner'], axis=1)
y = df['winner']

x = pd.get_dummies(x, ["team1","team2", "toss_winner", "toss_decision", "result"], drop_first = True)

x.head()

le = LabelEncoder()
y = le.fit_transform(y)
     

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=5)

# Model training:

model = RandomForestClassifier(n_estimators=100,min_samples_split=5,
                               max_features = "auto")

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)
accuracy
     
# Confusion Matrix:

confusion_matrix = confusion_matrix(y_test, y_pred
                                    )
cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
cm.plot()
plt.show()

c