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
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(bee_data.training_data, bee_data.training_targets, epochs=5)
    print("Scoring...")
    scores = model.evaluate(bee_data.validation_data, bee_data.validation_targets)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def main():
    # Read data
    bee_data = data("../Data/bee_data.csv", 5, "../Data/resized_bee_imgs") # Column 5 = Species
    bee_data.print()
    make_model(bee_data)

if __name__ == "__main__":
    main()
