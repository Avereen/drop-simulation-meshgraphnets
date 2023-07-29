
"""
Created on Fri Jul 14 16:47:24 2023

@author: alexv
"""


import pickle
path = "result/result0.pkl"

with open(path, 'rb') as f:
    data = pickle.load(f)