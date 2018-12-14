#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import imageio
from random import shuffle

# Data class designed to easily encapsulate data
# For our supervised learning Honey Bee Classifier
class data:
    def __init__(self, csv_path, target_feature, img_path):
        # Load data
        print("Loading data...")
        bee_targets, bee_imgs = self.load_from_array()
        self.bee_targets = bee_targets
        self.bee_imgs = bee_imgs
        # Get classes
        self.get_classes(bee_targets)
        # Split data
        self.split(bee_imgs, bee_targets)

    # Load from CSV and image file data
    def load_from_data(self, csv_path, img_path, target_feature):
        ######################
        # Load target values #
        ######################
        # Load targets (bee classification data)
        bee_csv = pd.read_csv(csv_path) # Read data csv
        bee_csv = bee_csv.values # Convert to numpy array
        bee_csv = bee_csv[np.argsort(bee_csv[:, 0])[::-1]] # Sort by first column = name of image
        shuffle(bee_csv) # Shuffle data
        # Load predictor data (images)
        bee_imgs = np.empty(len(bee_csv), dtype=object)
        for i in range(len(bee_csv)):
            bee_imgs[i] = imageio.imread(img_path+"/"+bee_csv[i,0]) # First item in bees is image name
        # Get just species targets
        bee_targets = bee_csv[:,target_feature]
        # Get classes
        self.get_classes(bee_targets)
        # One hot encode targets
        bee_num_targets = np.empty(len(bee_targets), dtype = int)
        for i in range(len(bee_targets)):
            bee_num_targets[i] = self.classes.index(bee_targets[i])
        bee_targets = bee_num_targets
        # Save array
        np.save("bee_targets_ndarray.npy", bee_targets)
        ##################################
        # Load predictor values (images) #
        ##################################
        # Convert imageio.core.util.array to numpy.ndarray [disgusting]
        np_bee_imgs = np.empty((len(bee_imgs), 77, 66, 3), dtype=int)
        # Iterate through the images
        for i in np.arange(len(bee_imgs)):
            # Build each image pixel by pixel
            for j in np.arange(77):
                for k in np.arange(66):
                    for l in np.arange(3):
                        np_bee_imgs[i,j,k,l] = bee_imgs[i][j,k,l]
        # Save array
        np.save("bee_imgs_ndarray.npy", np_bee_imgs)

    # Load from previously generated numpy arrays
    def load_from_array(self):
        np_bee_imgs = np.load("../Data/data_arrays/bee_imgs_ndarray.npy")
        bee_targets = np.load("../Data/data_arrays/bee_targets_ndarray.npy")
        return bee_targets, np_bee_imgs

    # Get classes for target values
    def get_classes(self, bee_targets):
        # Get unique types of species
        self.classes = []
        for item in bee_targets:
            if not item in self.classes:
                self.classes.append(item)
        print("Data classes found:", self.classes)

    # Split data into 60% training, 20% validation, and 20% testing
    def split(self, bee_imgs, bee_targets):
        print("Splitting data...")
        self.training_data = bee_imgs[:int(len(bee_imgs)*.6)]
        self.validation_data = bee_imgs[int(len(bee_imgs)*.6):int(len(bee_imgs)*.8)]
        self.full_training_data = bee_imgs[:int(len(bee_imgs)*.8)]
        self.testing_data = bee_imgs[int(len(bee_imgs)*.8):]
        self.training_targets = bee_targets[:int(len(bee_targets)*.6)]
        self.validation_targets = bee_targets[int(len(bee_targets)*.6):int(len(bee_targets)*.8)]
        self.full_training_targets = bee_targets[:int(len(bee_imgs)*.8)]
        self.testing_targets = bee_targets[int(len(bee_targets)*.8):]

    # To string
    def print(self):
        print("Printing info...")
        print("training_data: " + str(len(self.training_data)) + ", training_targets: " + str(len(self.training_targets)))
        print("validation_data: " + str(len(self.validation_data)) + ", validation_targets: " + str(len(self.validation_targets)))
        print("testing_data: " + str(len(self.testing_data)) + ", testing_targets: " + str(len(self.testing_targets)))
