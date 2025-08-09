import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
review_dataset = pd.read_csv('Review.data', sep='\t', names=column_names)

movie_title_dataset = pd.read_csv("Movie_Id_Titles")
movie_title_dataset.head()

df = pd.merge(review_dataset,movie_title_dataset,on='item_id')
df.head()

# EDA

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline

Let's create a ratings dataframe with average rating and number of ratings:
#average rating
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

#number of ratings
df.groupby('title')['rating'].count().sort_values(ascending=False).head()

ratings_data = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings_data.head()

Now set the number of ratings column:
ratings_data['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings_data.head()

Visualizing data with histograms:
plt.figure(figsize=(10,4))
ratings_data['num of ratings'].hist(bins=70);

plt.figure(figsize=(10,4))
ratings_data['rating'].hist(bins=70);

sns.jointplot(x='rating',y='num of ratings',data=ratings_data, alpha= 1);

#creating movie matrix
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()

ratings_data.sort_values('num of ratings',ascending=False).head(10)

ratings_data.head()

starwars_user_ratings = moviemat['Star Wars (1977)']  #sci-fi movie
liarliar_user_ratings = moviemat['Liar Liar (1997)']  #comedy
starwars_user_ratings.head()

We can then use corrwith() method to get correlations between two pandas series:

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings);

Let's clean this by removing NaN values and using a DataFrame instead of a series:

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

#highest correlated movies to star wars

corr_starwars.sort_values('Correlation',ascending=False).head(10)

#on the basis of corr + Number of ratings
corr_starwars = corr_starwars.join(ratings_data['num of ratings'])
corr_starwars.head()

#filter results where ratings > 100
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()

#similarly
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings_data['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()

