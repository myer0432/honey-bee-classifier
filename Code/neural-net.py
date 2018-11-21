import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

def read(filename):
    with open(filename) as file:
        data = file.readlines()
    data = [x.strip().split(",") for x in data[1:]]
    shuffle(data)
    targets = []
    for item in data:
        targets.append(item.pop(0))
    return (data[:int(len(data)*.1)], data[int(len(data)*.1):int(len(data)*.2)], data[int(len(data)*.2):],
        targets[:int(len(targets)*.1)], targets[int(len(targets)*.1):int(len(targets)*.2)], targets[int(len(targets)*.2):])

def main():
    testing_data, validation_data, training_data, testing_targets, validation_targets, training_targets = read("bee_data.csv")

    print(len(testing_data))
    print(testing_data[0])
    print(len(validation_data))
    print(validation_data[0])
    print(len(training_data))
    print(training_data[0])
    print(len(testing_targets))
    print(testing_targets[0])
    print(len(validation_targets))
    print(validation_targets[0])
    print(len(training_targets))
    print(training_targets[0])

if __name__ == "__main__":
    main()
