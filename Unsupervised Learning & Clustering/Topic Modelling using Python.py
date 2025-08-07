Topic Modelling
Topic Modelling is a Natural Language Processing technique to uncover hidden topics from text documents. It helps identify topics of the text documents to find relationships between the content of a text document and the topic.

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

data = pd.read_csv("articles.csv", encoding = 'latin1')
print(data.head())

As we are working on a Natural Language Processing problem, we need to clean the textual content by removing punctuation and stopwords. Here’s how we can clean the textual data:

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]
    # Join tokens to form preprocessed text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


data['Article'] = data['Article'].apply(preprocess_text)

Now we need to convert the textual data into a numerical representation. We can use text vectorization here:

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(data['Article'].values)

Now we will use an algorithm to identify relationships between the textual data to assign topic labels. We can use the Latent Dirichlet Allocation algorithm for this task. Latent Dirichlet Allocation (LDA) is a generative probabilistic algorithm used to uncover the underlying topics in a corpus of textual data. Let’s use the LDA algorithm to assign topic labels:

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(x)

topic_modelling = lda.transform(x)

topic_labels = np.argmax(topic_modelling, axis=1)
data['topic_labels'] = topic_labels

print(data.head())
