"""
This scripts contains 
data and functions useful for the other scripts
"""

import pickle

MOVIES = ['John Wick', 
          'Barbie', 
          'Men in Black',
          ]


with open('nmf_model_week10.pkl', 'rb') as file:
    nmf_model = pickle.load(file)

with open('cosin_recommender.pkl', 'rb') as file:
    cos_sim_model = pickle.load(file)