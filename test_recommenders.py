""" Write a program which checks that recommenders works as expected

we will use pytests
to instrall pytest run the terminal 
+pip or conda install pytest


TDD (test driven development cycle)"
0. Make a Hypothesis 
1. Write a test that fails
2. Change the code
3. Repeat if needed 

"""
from recommenders import random_recommender
from utils import MOVIES

def test_movies_are_strings():
    for movie in MOVIES: 
        assert isinstance(movie, str)


def test_for_two_movies():
    top2 = random_recommender(k=2)
    assert len(top2) == 2

def test_for_5_users():
    top5 = random_recommender(5)
    assert len(top5) ==5

    def test_return_0_if_k_is_10():
        top10 = random_recommender(10)
        assert len(top10) ==0