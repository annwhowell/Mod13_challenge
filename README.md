# Mod13_challenge: Venture Funding with Deep Learning

# Overview
This analysis predicts whether or not companies applying for venture capital funds from Alphabet Soup will be successful.
The analysis is based on 34,000 organizatons that have received over the years.
Using this data, three models were tested using binary classifier models with neural networks and deep learning.

# Data
The data of 34,000 companies was pulled from Path("./Resources/applicants_data.csv").
EIN and Name were dropped from the dataset
Categorical variables were created for Object type variables using OneHotEncoder

# Libraries and imports

'''
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
'''

# Outputs
Data from the three models and the weights were saved in the Resources file 

Path("./Resources/model.json")
Path("./Resources/A1_model.json")
Path("./Resources/A2_model.json")

"./Resources/AlphabetSoup.h5"
"./Resources/AlphabetSoup_A1.h5"
"./Resources/AlphabetSoup_A2.h5"

# Models

#Original Model: 
Had 2 layers (58 hidden nodes in layer 1 and 29 hidden nodes in layer 2)
Used relu activation for the layers
Used sigmoid activation for the output which had 1 output neuron
Compile used binary_crossentropy, adam optimizer and accuracy
It ran 50 epochs
Loss = .55
Accuracy = .73


#Alternative Model 1: 
Had 2 layers (40 hidden nodes in layer 1 and 20 hidden nodes in layer 2)
Used relu activation for the layers
Used sigmoid activation for the output which had 1 output neuron
Compile used binary_crossentropy, adam optimizer and accuracy
It ran 100 epochs
Loss = .55
Accuracy = .73


#Alternative Model 2: 
Had 3 layers (50 hidden nodes in layer 1 and 20 hidden nodes in layer 2 and 10 in layer 3)
Used tanh activation for the layers
Used sigmoid activation for the output which had 1 output neuron
Compile used binary_crossentropy, adam optimizer and accuracy
It ran 150 epochs
Loss = .55
Accuracy = .73

# Created By
Ann Howell with support from Rice University Fintech Bootcamp - March 2022

# License
MIT

