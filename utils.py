"""
This scripts contains 
data and functions useful for the other scripts
"""

import pickle

MOVIES = ['John Wick', 
          'Barbie', 
          'Men in Black',
          ]



with open('/Users/elizavetabugaeva/Documents/Spiced/weekly_milestones/week_10/week_10/NMF/nmf_model1.pkl', 'rb') as file:
    nmf_model = pickle.load(file)

with open('/Users/elizavetabugaeva/Documents/Spiced/weekly_milestones/week_10/cosin_recommender.pkl', 'rb') as file:
    cos_sim_model = pickle.load(file)