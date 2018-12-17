#!/usr/bin/env python3

from data_lib import *
from knn_lib import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

'''
1. Load labels and images
2. Find difference between all test instances and all training instances
3. Predict class with highest appearance in k samples
4. Find accuracy
5. Do hyperparameter search
'''

def main():

    # 1. Load labels and images
    csv_path = "../Data/bee_data.csv"
    imgs_path = "../Data/resized_bee_imgs"

    bee_csv, bee_imgs = load_data(csv_path, imgs_path, size_of_set=1)
    print("Set size:", bee_imgs.shape)

    # 2. Encode the classes
    bee_labels = np.asarray(bee_csv['subspecies'])
    for i in range(bee_labels.shape[0]):
        if bee_labels[i] == "-1":
            bee_labels[i] = -1
        elif bee_labels[i] == "Western honey bee":
            bee_labels[i] = 0
        elif bee_labels[i] == "Italian honey bee":
            bee_labels[i] = 1
        elif bee_labels[i] == "VSH Italian honey bee":
            bee_labels[i] = 2
        elif bee_labels[i] == "Carniolan honey bee":
            bee_labels[i] = 3
        elif bee_labels[i] == "Russian honey bee":
            bee_labels[i] = 4
        elif bee_labels[i] == "1 Mixed local stock 2":
            bee_labels[i] = 5


    '''
    -1 = -1
    0 = Western honey bee
    1 = Italian honey bee
    2 = VSH Italian honey bee
    3 = Carniolan honey bee
    4 = Russian honey bee
    5 = 1 Mixed local stock 2
    '''

    # 3. Split data into labeled and unlabeled sets
    known_samples, known_labels, unknown_samples, unknown_labels = split_data(bee_imgs, bee_labels, .8)
    np.save("../Data/train_val_samples", known_samples)
    np.save("../Data/train_val_labels", known_labels)
    np.save("../Data/test_samples", unknown_samples)
    np.save("../Data/test_labels", unknown_labels)
    print("Saved all sets4")
    print()

    print("CSV shape:", bee_csv.shape)
    print("--Images shape--")
    print("\tTotal:", bee_imgs.shape)
    print("\tSample:", bee_imgs[0].shape)
    print("Known Samples:", known_samples.shape, known_samples[0].shape)
    print("Known Labels:", known_labels.shape)
    print("Unknown Samples:", unknown_samples.shape, known_samples[0].shape)
    print("Unknown Labels:", unknown_labels.shape)

    '''
    predictions = knn(20, unknown_samples, known_samples, unknown_labels, known_labels)

    for i in range(predictions.shape[0]):
        predictions[i] = int(predictions[i])
        unknown_labels[i] = int(unknown_labels[i])

    print("Unknown Labels:", unknown_labels.shape, type(unknown_labels), type(unknown_labels[0]), unknown_labels[0])
    print("Predictions:", predictions.shape, type(predictions), type(predictions[0]), predictions[0])
    
    accuracy = accuracy_calc(unknown_labels, predictions)

    print("Accuracy:", accuracy)
    '''

    for master_loop in range(3):
        
        t = np.zeros(10, dtype=np.int32)
        for i in range(t.shape[0]):
            t[i] += 5*i
        print(t)
        hyper_predictions = hyperparameter_search(t, unknown_samples, known_samples, unknown_labels, known_labels)
        hyper_accuracies = hyperparameter_accuracies(unknown_labels, hyper_predictions)
        #graph_hyperparameter_search(t, hyper_accuracies)
        np.save("../Data/experiment_run_"+str(master_loop), hyper_predictions)
    

main()
