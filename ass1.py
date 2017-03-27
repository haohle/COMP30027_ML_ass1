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

M_FOLD = 10
K_NEIGHBOURS = 29

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
    result = method_func(instance1, instance2)
    return result

def get_neighbours(instance, training_data_set, k, method):
    '''
    return a list of (class, score) 2-tuples for each of the k best neighbours 
    for the given instnace from the test data set based on all of the instances
    from the test data set.
    arguments:
        instance: the instance we are trying to classify
        training_data_set: data set which will be fed to the instance
        method: str, the voting similarity (distance) methods used
    returns:
        list of (class, score) 2-tuples for each of the k best neighbours for
        the given instance
    '''
    # array approach - ignore for now
    # neighbours = np.zeros(shape=[k, 2]) # creates an array of k by 2 can also use np.empty
    # will need to convert neighbours back into a list before returning
    # return neighbours.tolist()
    
    raw = []
    for index, row in training_data_set.iterrows():
        temp_class = row['Rings']
        temp_score = compare_instance(instance, row, method)
        # fill up the list of classes and scores
        raw.append((temp_class, temp_score))

    # will sort them from shortest to longest distance then return a list of up to k items
    # the sorting part will probably take a while (not even sure what sorting algorithm is used)
    # might be best to use a heap instead and continuously update the top k items
    result = sorted(raw, key=lambda item: item[-1])[:k]
    return result

def predict_class(neighbours, method):
    '''
    return a predicted class label according to the given neighbours defined
    by a list of (class, score) 2-tuples & chosen voting methods
    arguments:
        neighbours: list of (class, score) 2-tuples for each of the best
                    neighbours for the given instance
        method: str, voting method to predict the class of the given instance
    returns:
        a predicted class label
    '''
    if method == "majority_voting":
        return majority_voting(neighbours)
    if method == "weighted_majority_ild":
        return weighted_majority(neighbours,
            distance_weighting_method=inverse_linear_distance)
    if method == "weighted_majority_id":
        return weighted_majority(neighbours,
            distance_weighting_method=inverse_distance,
            epsilon=0.5)

def evaluate(data_set,
        metric,
        distance_method="euclidean_distance",
        voting_method="majority_voting"):
    '''
    Evaluate the model by certain matric
    arguments:
        data_set: panda DataFrame, the data set
        metric: string, could be "accuracy", "precision" and "recall"
        distance_method: function to calculate similarity
            could be "euclidean_distance", "cosine_similarity"
        voting_method: function that to predict class by different voting method
            could be "majority_voting", "weighted_majority_ild",
                "weighted_majority_id"
    return: the calculated value of the evaluation metric based on dividing the
        given data set into training & test splits using your preferred
        evaluation strategy
    '''

    data_set = data_set.sample(frac=1)
    # split data into M_FOLD sets
    data_sets = [
        data_set[i:i + len(data_set) // M_FOLD]
        for i in range(0, len(data_set), len(data_set) // M_FOLD)]
    data_sets[-1] = data_sets[-1] + data_set[-len(data_set) % M_FOLD:]

    # should use i-th as testing data
    actual_classes = []
    predicted_classes = []
    for i in range(M_FOLD):
        curr_training = pd.DataFrame()
        for j in range(M_FOLD):
            # if the data is not testing data, add to training data
            if i != j:
                frames = [curr_training, data_sets[j]]
                curr_training = pd.concat(frames)

        # start training and testing
        testing_data = data_sets[i]
        
        for index, instance in testing_data.iterrows():
            neighbours = get_neighbours(
                instance,
                curr_training,
                K_NEIGHBOURS,
                distance_method)
            actual_classes.append(instance['Rings'])
            predicted_classes.append(
                predict_class(neighbours, voting_method))

    # evaluate the model
    metric2func = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    return metric2func[metric](actual_classes, predicted_classes)

@profile
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
    s = (SEX2NUM[instance1[0]] - SEX2NUM[instance2[0]]) ** 2
    for i in range(1, len(instance1) - 1):
        s += (instance1[i] - instance2[i]) ** 2
    return s ** 0.5

def cosine_similarity(instance1, instance2):
    '''
    Find the similarity by using cosine similarity
    nominal data here will be assigned to 1, 2, 3
    arguments:
        instance1: iterable, one of the instances to be compared
        instance2: iterable, another of the instances to be compared
    return: the cosine of the angle between the vector of instance 1 and 2
    '''
    # exclude the last item in the vector as it is the class
    sqlength1 = SEX2NUM[instance1[0]] ** 2
    sqlength2 = SEX2NUM[instance2[0]] ** 2
    numerator = SEX2NUM[instance1[0]] * SEX2NUM[instance2[0]]
    for i in range(1, len(instance1) - 1):
        numerator += instance1[i] * instance2[i]
        sqlength1 += instance1[i] ** 2
        sqlength2 += instance2[i] ** 2
    # ** 0.5 is sqrt
    denominator = (sqlength1 ** 0.5) * (sqlength1 ** 0.5)
    return numerator / denominator

def majority_voting(neighbours):
    '''
    Calculates the majority (if any) of classes from the given neighbours
    arguments:
        neighbours: is a list of (class, scores) 2-tuples. only need to make
                    use of class for this voting method
    return: a string with the class name for the majority found in the list
            of neighbours
    '''
    # only need the class labels, not the score for majority voting
    # will therefore ignore the scores all together
    classLabels = [i[0] for i in neighbours]

    # as a draw will never occue if we only use odd values for k
    return max(set(classLabels), key=classLabels.count)

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

def majority_voting(neighbours):
    '''
    Calculates the majority (if any) of classes from the given neighbours
    arguments:
        neighbours: is a list of (class, scores) 2-tuples. only need to make
                    use of class for this voting method
    return: a string with the class name for the majority found in the list
            of neighbours
    '''
    # only need the class labels, not the score for majority voting
    # will therefore ignore the scores altogether
    classLabels = [i[0] for i in neighbours]

    # a draw will never occur if only odd values for k are used
    return max(set(classLabels), key=classLabels.count)

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

def precision(actual_classes, predicted_classes):
    '''
    Find the precision of the model
    arguments:
        actual_classes: list of classes of the actual value, supervised learning
        prediced_classes: list of predicted classes, order as actual_classes
    return:
        dict:
            key: class
            value: float that represent the precision
    '''
    predicted_count = dd(int)
    tp_count = dd(int)
    length = len(actual_classes)
    for i in range(length):
        predicted_count[predicted_classes[i]] += 1
        if(predicted_classes[i] == actual_classes[i]):
            tp_count[predicted_classes[i]] += 1
    result = {}
    for label in predicted_count:
        result[label] = tp_count[label] / predicted_count[label]
    return result

def recall(actual_classes, predicted_classes):
    '''
    Find the recall of the model
    arguments:
        actual_classes: list of classes of the actual value, supervised learning
        prediced_classes: list of predicted classes, order as actual_classes
    return:
        dict:
            key: class
            value: float that represent the recall
    '''
    actual_count = dd(int)
    tp_count = dd(int)
    length = len(actual_classes)
    for i in range(length):
        actual_count[actual_classes[i]] += 1
        if(predicted_classes[i] == actual_classes[i]):
            tp_count[predicted_classes[i]] += 1
    result = {}
    for label in actual_count:
        result[label] = tp_count[label] / actual_count[label]
    return result

def accuracy(actual_classes, predicted_classes):
    '''
    Find the classification accuracy of the model
    accuracy = (total correct predictions) / (total classes)
    arguments:
        actual_classes: list of classes of the actual value, supervised learning
        predicted_classes: list of predicted classes, order as actual_classes
    return:
        dict:
            key: class
            value: float that represents the precision
    '''
    total_classes = len(actual_classes)

    for i in range(length):
        if (predicted_classes[i] == actual_classes[i]):
            correct_count += 1

    return float(correct_count / total_classes)

if __name__ == "__main__":
    #pass
    print(evaluate(preprocess_data('./data/abalone.data'), "accuracy"))
