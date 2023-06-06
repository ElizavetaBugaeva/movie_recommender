"""
This scripts contains 
data and functions useful for the other scripts
"""
import numpy as np
from sklearn.decomposition import NMF 
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import json


movies= pd.read_csv('ml-latest-small/movies.csv' , sep = ',')
ratings= pd.read_csv('ml-latest-small/ratings.csv' , sep = ',')
links = pd.read_csv('ml-latest-small/links.csv' , sep = ',')
tags = pd.read_csv('ml-latest-small/tags.csv' , sep = ',')
query = 'user_query.json'
df_R = pd.read_csv('user_rating.csv')
title = pd.read_csv('processed_titles')



def convert_json_to_query(query):
    try:
        data = json.loads(query)
        new_user_query = ""
        for title, rating in data.items():
            movie_id = int(input("Enter movie ID for '{}': ".format(title)))
            new_user_query += "{}:{} ".format(movie_id, rating)
        return query.strip()
    except json.JSONDecodeError as e:
        print("Invalid JSON input:", e)
    except ValueError as e:
        print("Invalid movie ID:", e)

new_user_query = convert_json_to_query(query)

MOVIES = [
    "Men in Black",
    "Inglorious Bastards", 
    "My names is Nobody",
    "John Wick four",
    "Pocahontas", 
    "The Fast and Furious (8)", 
    "Tom and Jerry",
    "5"
]

nmf_model = 'nmf_model.pkl'
cos_sim_model = 'cosim_recommender.pkl'

def recommend_nmf(query, model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    
    # 1. construct new_user-item dataframe given the query
    rated_movies = set(query.keys())
    recommendations = ratings[ratings.index.isin(rated_movies)].sort_values(by='rating', ascending=False).head(k)
    new_user_dataframe = pd.DataFrame(new_user_query,columns = df_R.columns, index = ['new user'])
    new_user_imputed = new_user_dataframe.fillna(df_R.mean())

    # 2. scoring
    # calculate the score with the NMF model
    P_new_user_matrix = model.transform(new_user_imputed)
    Q_matrix = model.components_
    R_hat_new_user_matrix = np.dot(P_new_user_matrix, Q_matrix)
    
    # 3. Ranking
    # Filter out movies already seen by the user
    R_hat_new_user_filtered = pd.DataFrame(R_hat_new_user_matrix, columns=df_R.columns)
    R_hat_new_user_filtered = R_hat_new_user_filtered.loc[:, ~R_hat_new_user_filtered.columns.isin(rated_movies)]
    
    # Return the top-k highest rated movie ids or titles
    recommended = R_hat_new_user_filtered.iloc[0].nlargest(k)
    recommended_movies = title[title['movieId'].isin(recommended.index)]

    return recommended_movies[['movieId', 'title']].values.tolist()

# collaborative filtering = look at ratings only!
def recommend_neighborhood(query, model, ratings, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    # construct a user vector
    new_user_dataframe = pd.DataFrame(new_user_query,columns = df_R.columns, index = ['new user'])
    new_user_imputed = new_user_dataframe.fillna(df_R.mean())
   
    # 2. scoring
    # find n neighbors
    similarity_scores, neighbor_ids = cos_sim_model.kneighbors(
    new_user_imputed,
    n_neighbors=5,
    return_distance=True)
    neighborhood = df_R.iloc[neighbor_ids[0]]

    # 3. ranking
    # filter out movies allready seen by the user:
    # 1 - Convert keys in new_user_query to strings
    query_keys = [str(key) for key in new_user_query.keys()]
    # 2 - Filter out the movies already seen
    neighborhood_filtered = neighborhood.drop(query_keys, axis=1)

    # return the top-k highst rated movie ids or titles
    df_score = neighborhood_filtered.sum()
    df_score_ranked = df_score.sort_values(ascending=False).index.astype(int).tolist()[:k]
    recommendations = df_score_ranked[:k]
    recommended_movies = title[title['movieId'].isin(recommendations)]
    
    return recommended_movies[['movieId', 'title']].values.tolist()