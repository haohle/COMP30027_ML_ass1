import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict as dd

HEADER = ["Sex",
	"Length",
	"Diameter",
	"Height",
	"Whole weight",
	"Shucked weight",
	"Viscera weight",
	"Shell weight",
	"Rings"]
SEX2NUM = {
    "M": 1,
    "F": 2,
    "I": 3,
}


def preprocess_data(filename):
    '''
    opens the file given by String 'filename' returns a dataset
    (with suitable class labels) made up of instances of the file, 1 per line
    '''
    print("Reading in: {}".format(filename))

    raw = pd.read_csv(filename, names=HEADER)
    raw.dropna(how='any')   # removes any instances with missing data (if any)

    return raw

def compare_instance(instance1, instance2, method):
    '''
    arguments:
    	instance1: iterable, one of the instances to be compared
    	instance2: iterable, another of the instances to be compared
    	method: 
    returns: a score based on calculating the similarity (or distane) between
    	two given instances according to the similarity (or distance) metric
    	defined by the string method specify the values that method
    '''
    method_func = globals().get(method)
    if not method_func:
    	raise NotImplementedError(
    		"Method {} not implemented".format(method))
    return method_func(instance1, instance2)

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

def euclidean_distance(instance1, instance2):
    '''
    Find the similarity (distance) by using euclidean distance.
    Nominal data will be assigned to 1, 2, 3 (this is prevalent for the Sex 
    attribute).
    arguments:
        instance1: iterable, one of the instances to be compared
        instance2: iterable, another of the instances to be comapred
    returned: the euclidean distance between the vector of instance 1 and 2
    '''
    instnace1 = np.array((SEX2NUM[instance1[0]],) + instance1[1:])
    instnace1 = np.array((SEX2NUM[instance2[0]],) + instance2[1:])
    return np.linalg.norm(instnace1 - instance2)

def cosine_similarity(instance1, instance2):
    '''
    Find the similarity by using cosine similarity
    nominal data here will be assigned to 1, 2, 3
    arguments:
        instance1: iterable, one of the instances to be compared
        instance2: iterable, another of the instances to be compared
    return: the cosine of the angle between the vector of instance 1 and 2
    '''
    instance1 = np.array((SEX2NUM[instance1[0]],) + instance1[1:])
    instance2 = np.array((SEX2NUM[instance2[0]],) + instance2[1:])
    return np.dot(instance1, instance2) / \
        (np.linalg.norm(instance1) * np.linalg.norm(instance2))

def inverse_linear_distance(distances):
    '''
    This is a method for weighted_majority
    arguments:
        distances: all the distances from the test instance
    return: a list of weights, same order as distances
    '''
    # avoid empty list to crash the program
    if not distances:
        return []

    # finding d1 and dk,
    # d1 is the nearest distance, initialised to very far
    # dk is the furthest distance, initialised to negative
    d1 = float('Inf')
    dk = -float('Inf')
    for distance in distances:
        if distance < d1:
            d1 = distance
        if distance > dk:
            dk = distance

    # difference between the furthest and nearest
    # if differnce is 0, will assume that is equal weight,
    # just return all 1s, cause need to avoid divide by 0 error
    furthest_nearest_distance = dk - d1
    if furthest_nearest_distance == 0:
        return [1 for i in range(len(distances))]

    # find the actual weights
    weights = []
    for distance in distances:
        if distance == d1:
            weights.append((i, 1))
        else:
            weights.append((i, (dk - dj) / furthest_nearest_distance))

    return weights

def inverse_distance(distances, epsilon=0.5):
    '''
    This is a method for weighted_majority
    arguments:
        distances: all the distances from the test instance
        epsilon: default to 0.5, an offset to denominator
    return: a list of weights, same order as distances
    '''
    return [1 / (distance + epsilon) for distance in distances]

def weighted_majority(neighbours,
        distance_weighting_method=inverse_distance,
        epsilon=0.5):
    '''
    Find the weighted majority
    This function support different distance weighting method
    arguments:
        neighbours: the list of nearest neighbours
        distance_method: the function that calculate the distance
        distance_weighting_method: the method for weighting the distance
        epsilon: mainly just for inverse_distance
    return: the predicted class base on weighted majority
    '''
    # just distances of all neighbours,
    # just not to less the weighting function to mess up the data
    distances = \
        [distance for instance, distance in neighbours]

    # finding the weights
    if distance_weighting_method == inverse_distance:
        weights = distance_weighting_method(distances, epsilon)
    else:
        weights = distance_weighting_method(distances)
    
    # start counting up for classes now
    class_counts = dd(float)
    for i in range(len(neighbours)):
        class_counts[neighbours[i][0]] += weights[i]

    # after counting up, find the most counted class
    max_count = -1
    pred_class = None
    for label, count in class_counts.items():
        if count > max_count:
            max_count = count
            pred_class = label
    
    return pred_class

if __name__ == "__main__":
	pass
