import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
#load dataset
books = pd.read_csv('books dataset/BX-Books.csv',sep=';',error_bad_lines=False,encoding='latin-1')
books.head()

books.shape

#select only columns needed
books = books[['ISBN','Book-Title','Book-Author','Year-Of-Publication','Publisher','Image-URL-L']]
books.head(2)

books.rename(columns={
    'Book-Title':'Title',
    'Book-Author':'Author',
    'Year-Of-Publication':'year',
    'Image-URL-L' : 'image_url'},
    inplace =True
)

user = pd.read_csv('books dataset/BX-Users.csv',sep=';',error_bad_lines=False,encoding='Latin-1')
user.head()

user.shape

ratings = pd.read_csv('books dataset/BX-Book-Ratings.csv',sep=';',error_bad_lines=False,encoding='Latin-1')
ratings.head()

ratings.rename(columns={'User-ID':'user_id','Book-Rating':'rating'}, inplace=True)

#shapes of all dataset
print(books.shape)
print(user.shape)
print(ratings.shape)

ratings.head()

#ratings counts
rating_val_count = ratings['user_id'].value_counts()
rating_val_count

#filtering for user_id with more than 200 ratings
ratings200plus = ratings['user_id'].value_counts() > 200
ratings200plus

#get the user_ids
index_ratings200plus = ratings200plus[ratings200plus].index
index_ratings200plus

ratings =ratings[ratings['user_id'].isin(index_ratings200plus)]
ratings.head()

ratings_with_books = ratings.merge(books, on='ISBN')
ratings_with_books.head(3)

num_rating = ratings_with_books.groupby('Title')['rating'].count().reset_index()
num_rating.head()

num_rating.rename(columns={'rating':'num_of_rating'},inplace=True)
num_rating.head(3)

final_rating = ratings_with_books.merge(num_rating, on='Title')
final_rating.head(3)

final_rating_df = final_rating[final_rating['num_of_rating'] >= 50]
final_rating_df.head(3)

final_rating_df.drop_duplicates(['user_id','Title','year'],inplace=True)

final_rating_df.shape

#create a pivot_table 

pivot_df = final_rating_df.pivot_table(
    columns='user_id',index='Title',values='rating')

pivot_df

pivot_df.fillna(0,inplace=True)
pivot_df

from scipy.sparse import csr_matrix

book_sparse = csr_matrix(pivot_df)
book_sparse

from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm='brute')

model.fit(pivot_df)

distance, suggestion = model.kneighbors(
    pivot_df.iloc[237,:].values.reshape(1,-1),n_neighbors=6)
distance

for i in range(len(suggestion)):
    print(pivot_df.index[suggestion[i]])

books_name = pivot_df.index
import pickle
pickle.dump(model, open('artifacts/model.pkl', 'wb'))
pickle.dump(books_name, open('artifacts/books_name.pkl', 'wb'))
pickle.dump(final_rating_df, open('artifacts/final_rating_df.pkl', 'wb'))
pickle.dump(pivot_df, open('artifacts/pivot_df.pkl', 'wb'))
def recommend_book(book_name):
    book_id = np.where(pivot_df.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(pivot_df.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
    
    for i in range(len(suggestion)):
        books = pivot_df.index[suggestion[i]]
        for j in books:
            print(j)
recommend_book('Jacob Have I Loved')

import ipywidgets 

widgets = ipywidgets.Dropdown(
    options = books_name,
    value = '2nd Chance',
    description = 'Movie Title:',
    disabled = False
)

ipywidgets.interact(recommend_book, book_name=widgets)

