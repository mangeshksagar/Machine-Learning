#%%
import numpy as np
import pandas as pd
#%%
#Reading users file:
users_cols = ['user_id', 'age','gender', 'occupation', 'zip_code']
users_df = pd.read_csv('u.user', sep='|', names=users_cols,encoding='latin-1')

#%%
#Reading ratings file:-
rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_df = pd.read_csv('u.data', sep='\t', names=rating_cols,encoding='latin-1')
#%%
#Reading Rating Test & Train Files:-
ratings_train_df = pd.read_csv('ua.base', sep='\t', names=rating_cols, encoding='latin-1')
ratings_test_df = pd.read_csv('ua.test', sep='\t', names=rating_cols, encoding='latin-1')
#%%
##Reading items file:-
items_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items_df = pd.read_csv('u.item', sep='|', names=items_cols,encoding='latin-1')
#%%
#Checking Users Files Size:-
print(users_df.shape)
users_df.head()
#%%
#Checking Rating Files Size:-
print(ratings_df.shape)
ratings_df.head()
#%%
#Checking Items Size:-
print(ratings_df.shape)
ratings_df.head()
#%%
ratings_train_df.shape, ratings_test_df.shape
ratings_train_df.head(),ratings_test_df
#%%
no_users = ratings_df.user_id.unique().shape[0]
no_items = ratings_df.movie_id.unique().shape[0]
#%%
#Making Matrix Of Dimensions equal to userid
matrix = np.zeros((no_users, no_items))
#%%
for line in ratings_df.itertuples():
    matrix[line[1]-1, line[2]-1] = line[3]
#%%
#Calculating Similarity Measures:-
from sklearn.metrics.pairwise import pairwise_distances 
user_similarity = pairwise_distances(matrix, metric='cosine')
item_similarity = pairwise_distances(matrix.T, metric='cosine')
#%%
def predict(ratings_df, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings_df.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings_df - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings_df.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
#%%
user_prediction = predict(matrix, user_similarity, type='user')
item_prediction = predict(matrix, item_similarity, type='item')
#%%
ratings_train_df
#%%
import turicreate
train_data = turicreate.SFrame(ratings_train_df)
test_data = turicreate.SFrame(ratings_test_df)
#%%
#Using Turicate Model to predict:-
popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
#%%
popularity_model
#%%