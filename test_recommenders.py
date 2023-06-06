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

from utils import MOVIES

def test_movies_are_strings():
    for movie in MOVIES: 
        assert isinstance(movie, str)

