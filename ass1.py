import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def preprocess_data(filename):
    '''
    opens the file given by String 'filename' returns a dataset
    (with suitable class labels) made up of instances of the file, 1 per line
    '''
    print("Reading in: {}".format(filename))

    raw = pd.read_csv(filename)
    raw.dropna(how='any')   # removes any instances with missing data (if any)

    return raw

def compare_instance(instance1, instance2, method):
    '''
    returns a score based on calculating the similarity (or distane) between two
    given instances according to the similarity (or distance) metric defined by
    the string method specify the values that method
    '''
    pass

def get_neighbours(instance, training_data_set, k, methd):
    '''
    return a list of (class, score) 2-tuples for each of the k best neighbours 
    for the given instnace from the test data set based on all of the instances
    from the test data set
    '''
    pass


def predict_class(neighbours, method):
    '''
    return a predicted class label according to the given neighbours defined
    by a list of (class, score) 2-tuples & chosen voting methods
    '''
    pass

def evaluate(data_set, metric):
    '''
    return the calculated value of the evaluation metric based on dividing the
    given data set into training & test splits using your preferred evaluation
    strategy
    '''
    pass

if __name__ == "__main__":
    '''
    '''
    pass
