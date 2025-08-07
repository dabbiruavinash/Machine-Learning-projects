The problem of consumer complaint classification is based on Natural Language Processing and Multiclass Classification. To solve this problem, we needed a dataset containing complaints reported by consumers.

I found an ideal dataset for this task that contains data about:

The nature of the complaint reported by the consumer
The Issue mentioned by the consumer
The complete description of the complaint of the consumer

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import nltk
import re
from nltk.corpus import stopwords
import string

data = pd.read_csv("/content/drive/MyDrive/consumercomplaints.csv")
print(data.head())

data = pd.read_csv("consumercomplaints.csv")

data = data.drop("Unnamed: 0",axis=1)

print(data.isnull().sum())

data = data.dropna()

print(data["Product"].value_counts())

Training Consumer Complaint Classification Model:

nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["Consumer complaint narrative"] = data["Consumer complaint narrative"].apply(clean)

Now, let’s split the data into training and test sets:

data = data[["Consumer complaint narrative", "Product"]]
x = np.array(data["Consumer complaint narrative"])
y = np.array(data["Product"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42)

Now, let’s train the Machine Learning model using the Stochastic Gradient Descent classification algorithm:

sgdmodel = SGDClassifier()
sgdmodel.fit(X_train,y_train)

Now, let’s use our trained model to make predictions:

user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = sgdmodel.predict(data)
print(output)

user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = sgdmodel.predict(data)
print(output)

