import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import matplotlib.pyplot as plt
from data import data

def make_model(bee_data):
    print("Training model...")
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(1, activation=tf.nn.relu))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(bee_data.training_data, bee_data.training_targets, epochs=150, batch_size=10)
    print("Scoring...")
    scores = model.evaluate(bee_data.validation_data, bee_data.validation_targets)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def main():
    # Read data
    bee_data = data("../Data/bee_data.csv", 5, "../Data/bee_imgs") # Column 5 = Species
    bee_data.print()
    print(bee_data.training_data)
    make_model(bee_data)

if __name__ == "__main__":
    main()
