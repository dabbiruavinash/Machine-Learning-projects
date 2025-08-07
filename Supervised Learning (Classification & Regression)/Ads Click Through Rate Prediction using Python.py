import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
pio.templates.default = "plotly_white"

data = pd.read_csv("ad_10000records.csv")
print(data.head())

data["Clicked on Ad"] = data["Clicked on Ad"].map({0: "No", 1: "Yes"})

Click Through Rate Analysis:

fig = px.box(data, 
             x="Daily Time Spent on Site",  
             color="Clicked on Ad", 
             title="Click Through Rate based Time Spent on Site", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

From the above graph, we can see that the users who spend more time on the website click more on ads. Now let’s analyze the click-through rate based on the daily internet usage of the user:

fig = px.box(data, 
             x="Daily Internet Usage",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Daily Internet Usage", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

From the above graph, we can see that the users with high internet usage click less on ads compared to the users with low internet usage. Now let’s analyze the click-through rate based on the age of the users:

fig = px.box(data, 
             x="Age",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Age", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

From the above graph, we can see that users around 40 years click more on ads compared to users around 27-36 years old. Now let’s analyze the click-through rate based on the income of the users:

fig = px.box(data, 
             x="Area Income",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Income", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

Calculating CTR of Ads:

Now let’s calculate the overall Ads click-through rate. Here we need to calculate the ratio of users who clicked on the ad to users who left an impression on the ad. So let’s see the distribution of users:

data["Clicked on Ad"].value_counts()

So 4917 out of 10000 users clicked on the ads. Let’s calculate the CTR:

click_through_rate = 4917 / 10000 * 100
print(click_through_rate)

Click Through Rate Prediction Model:

data["Gender"] = data["Gender"].map({"Male": 1, 
                               "Female": 0})

x=data.iloc[:,0:7]
x=x.drop(['Ad Topic Line','City'],axis=1)
y=data.iloc[:,9]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.2,
                                           random_state=4)

Now let’s train the model using the random forecast classification algorithm:

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x, y)

Now let’s have a look at the accuracy of the model:

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,y_pred))

Now let’s test the model by making predictions:

print("Ads Click Through Rate Prediction : ")
a = float(input("Daily Time Spent on Site: "))
b = float(input("Age: "))
c = float(input("Area Income: "))
d = float(input("Daily Internet Usage: "))
e = input("Gender (Male = 1, Female = 0) : ")

features = np.array([[a, b, c, d, e]])
print("Will the user click on ad = ", model.predict(features))

