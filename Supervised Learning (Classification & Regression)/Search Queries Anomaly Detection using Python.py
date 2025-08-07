Search Queries Anomaly Detection: Process We Can Follow
Search Queries Anomaly Detection is a technique to identify unusual or unexpected patterns in search query data. Below is the process we can follow for the task of Search Queries Anomaly Detection:

Gather historical search query data from the source, such as a search engine or a website’s search functionality.
Conduct an initial analysis to understand the distribution of search queries, their frequency, and any noticeable patterns or trends.
Create relevant features or attributes from the search query data that can aid in anomaly detection.
Choose an appropriate anomaly detection algorithm. Common methods include statistical approaches like Z-score analysis and machine learning algorithms like Isolation Forests or One-Class SVM.
Train the selected model on the prepared data.
Apply the trained model to the search query data to identify anomalies or outliers.

import pandas as pd
from collections import Counter
import re
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

queries_df = pd.read_csv("Queries.csv")
print(queries_df.head())

Exploratory Data Analysis:

print(queries_df.info())

Now, let’s convert the CTR column from a percentage string to a float:

# Cleaning CTR column
queries_df['CTR'] = queries_df['CTR'].str.rstrip('%').astype('float') / 100

Now, let’s analyze common words in each search query:

# Function to clean and split the queries into words
def clean_and_split(query):
    words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
    return words

# Split each query into words and count the frequency of each word
word_counts = Counter()
for query in queries_df['Top queries']:
    word_counts.update(clean_and_split(query))

word_freq_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])

# Plotting the word frequencies
fig = px.bar(word_freq_df, x='Word', y='Frequency', title='Top 20 Most Common Words in Search Queries')
fig.show()

Now, let’s have a look at the top queries by clicks and impressions:

# Top queries by Clicks and Impressions
top_queries_clicks_vis = queries_df.nlargest(10, 'Clicks')[['Top queries', 'Clicks']]
top_queries_impressions_vis = queries_df.nlargest(10, 'Impressions')[['Top queries', 'Impressions']]

# Plotting
fig_clicks = px.bar(top_queries_clicks_vis, x='Top queries', y='Clicks', title='Top Queries by Clicks')
fig_impressions = px.bar(top_queries_impressions_vis, x='Top queries', y='Impressions', title='Top Queries by Impressions')
fig_clicks.show()
fig_impressions.show()

Now, let’s analyze the queries with the highest and lowest CTRs:

# Queries with highest and lowest CTR
top_ctr_vis = queries_df.nlargest(10, 'CTR')[['Top queries', 'CTR']]
bottom_ctr_vis = queries_df.nsmallest(10, 'CTR')[['Top queries', 'CTR']]

# Plotting
fig_top_ctr = px.bar(top_ctr_vis, x='Top queries', y='CTR', title='Top Queries by CTR')
fig_bottom_ctr = px.bar(bottom_ctr_vis, x='Top queries', y='CTR', title='Bottom Queries by CTR')
fig_top_ctr.show()
fig_bottom_ctr.show()

Now, let’s have a look at the correlation between different metrics:

# Correlation matrix visualization
correlation_matrix = queries_df[['Clicks', 'Impressions', 'CTR', 'Position']].corr()
fig_corr = px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix')
fig_corr.show()

In this correlation matrix:

Clicks and Impressions are positively correlated, meaning more Impressions tend to lead to more Clicks.
Clicks and CTR have a weak positive correlation, implying that more Clicks might slightly increase the Click-Through Rate.
Clicks and Position are weakly negatively correlated, suggesting that higher ad or page Positions may result in fewer Clicks.
Impressions and CTR are negatively correlated, indicating that higher Impressions tend to result in a lower Click-Through Rate.
Impressions and Position are positively correlated, indicating that ads or pages in higher Positions receive more Impressions.
CTR and Position have a strong negative correlation, meaning that higher Positions result in lower Click-Through Rates.

Detecting Anomalies in Search Queries

Now, let’s detect anomalies in search queries. You can use various techniques for anomaly detection. A simple and effective method is the Isolation Forest algorithm, which works well with different data distributions and is efficient with large datasets:

from sklearn.ensemble import IsolationForest

# Selecting relevant features
features = queries_df[['Clicks', 'Impressions', 'CTR', 'Position']]

# Initializing Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01)  # contamination is the expected proportion of outliers

# Fitting the model
iso_forest.fit(features)

# Predicting anomalies
queries_df['anomaly'] = iso_forest.predict(features)

# Filtering out the anomalies
anomalies = queries_df[queries_df['anomaly'] == -1]

Here’s how to analyze the detected anomalies to understand their nature and whether they represent true outliers or data errors:

print(anomalies[['Top queries', 'Clicks', 'Impressions', 'CTR', 'Position']])

