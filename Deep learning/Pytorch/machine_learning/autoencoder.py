Importing
the
libraries
import numpy as np
import pandas as pd

# Importing the dataset
movies = pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Boltzmann_Machines/Boltzmann_Machines/ml-1m/movies.dat',
                     sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Boltzmann_Machines/Boltzmann_Machines/ml-1m/users.dat',
                    sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Boltzmann_Machines/Boltzmann_Machines/ml-1m/ratings.dat',
                      sep='::', header=None, engine='python', encoding='latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv(
    '/home/mayank-s/PycharmProjects/Datasets/Boltzmann_Machines/Boltzmann_Machines/ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Boltzmann_Machines/Boltzmann_Machines/ml-100k/u1.test',
                       delimiter='\t')
test_set = np.array(test_set, dtype='int')
