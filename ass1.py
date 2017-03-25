import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

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

    df = pd.DataFrame(raw)  # converts the raw csv into a dataframe

    return df

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

def get_neighbours(instance, training_data_set, k, method):
    '''
    return a list of (class, score) 2-tuples for each of the k best neighbours 
    for the given instnace from the test data set based on all of the instances
    from the test data set.
    arguments:
        instance: the instance we are trying to classify
        training_data_set: data set which will be fed to the instance
        method: the voting similarity (distance) methods used
    returns:
        list of (class, score) 2-tuples for each of the k best neighbours for
        the given instance
    '''
    pass


def predict_class(neighbours, method):
    '''
    return a predicted class label according to the given neighbours defined
    by a list of (class, score) 2-tuples & chosen voting methods
    arguments:
        neighbours: list of (class, score) 2-tuples for each of the best
                    neighbours for the given instance
        method: voting method to predict the class of the given instance
    returns:
        a predicted class label
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

if __name__ == "__main__":
    #pass
    df = preprocess_data('/data/abalone.data')
