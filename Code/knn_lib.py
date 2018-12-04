#!/usr/bin/env python3

import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

# 0
def PCA():
    a = 0

# 0
def kmc(samples):
    a = 0

# 1
def knn(k, unknown_samples, known_samples, unknown_labels, known_labels):
    final_distance = np.zeros((unknown_samples.shape[0], known_samples.shape[0]), dtype=np.float32)

    for i in range(unknown_samples.shape[0]):
        for j in range(known_samples.shape[0]):

            distance_1 = np.subtract(unknown_samples[i],known_samples[j])
            distance_2 = np.absolute(distance_1)
            final_distance[i][j] = np.sum(distance_2)
        print("Test sample "+str(i)+" done")

    ### Sorting, tricky stuff ###
    sorted_labels = np.array([known_labels,]*unknown_samples.shape[0])
    sorted_final_distance = np.array(final_distance)
    for i in range(final_distance.shape[0]):
        sorted_args = np.argsort(final_distance[i], axis=0)
        sorted_final_distance[i] = final_distance[i,sorted_args]
        sorted_labels[i] = sorted_labels[i,sorted_args]
    print("Sorted final distance:", sorted_final_distance.shape)
    print("Sorted labels:", sorted_labels.shape)
    print("Sorted args:", sorted_args.shape)

    '''
    t = np.linspace(0, final_distance.shape[1], final_distance.shape[1])
    plt.plot(t, sorted_final_distance[0,:])
    plt.show()
    '''

    print(type(sorted_labels))
    print(type(sorted_labels[0,0]))
    print(sorted_labels[0,0])
    # 7 possible classes
    bins = np.zeros((unknown_samples.shape[0],7), dtype=np.int32)
    predictions = np.zeros(unknown_samples.shape[0], dtype=np.int32)
    for i in range(sorted_final_distance.shape[0]):
        if k == 0:
            k = 1
        for j in range(k):
            label = sorted_labels[i,j]
            if label == -1:
                bins[0] += 1
            elif label == 0:
                bins[1] += 1
            elif label == 1:
                bins[2] += 1
            elif label == 2:
                bins[3] += 1
            elif label == 3:
                bins[4] += 1
            elif label == 4:
                bins[5] += 1
            elif label == 5:
                bins[6] += 1
    
    argmaxes = np.argmax(bins, axis=1)
    
    for i in range(argmaxes.shape[0]):
        predictions[i] = argmaxes[i]-1
    

    return predictions
    
def accuracy_calc(labels, predictions):
    accuracy = 0
    for i in range(labels.shape[0]):
        if labels[i] == predictions[i]:
            accuracy += 1

    accuracy = accuracy/labels.shape[0]

    return accuracy

# 1
def hyperparameter_search(t, unknown_samples, known_samples, unknown_labels, known_labels):
    
    hyper_predictions = np.empty(t.shape[0], dtype=np.object)
    for i in range(t.shape[0]):
        print("Hyperparameter")
        print("\tIndex:", i)
        print("\tK:", t[i])
        hyper_predictions[i] = knn(int(t[i]), unknown_samples, known_samples, unknown_labels, known_labels)

    return hyper_predictions

def hyperparameter_accuracies(labels, predictions):
    accuracies = np.zeros(predictions.shape, dtype=np.float32)
    for i in range(predictions.shape[0]):
        accuracies[i] = accuracy_calc(labels, predictions[i])

    return accuracies

# 1
def graph_hyperparameter_search(t, results):
    plt.plot(t, results)
    plt.xlabel("k Parameter")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. k Parameter")
    plt.show()
