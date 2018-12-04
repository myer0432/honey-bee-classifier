#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import imageio
from random import shuffle

class data:
    def __init__(self, csv_path, target_feature, img_path):
        # Load data
        print("Loading data...")
        bee_csv, bee_imgs = self.load_data(csv_path, img_path)
        # Get just species targets
        bee_targets = bee_csv[:,target_feature]
        # Get classes
        self.get_classes(bee_targets)
        # Split data
        self.split(bee_imgs, bee_targets)

    def load_data(self, csv_path, img_path):
        # Load targets (bee classification data)
        bee_csv = pd.read_csv(csv_path) # Read data csv
        bee_csv = bee_csv.values # Convert to numpy array
        bee_csv = bee_csv[np.argsort(bee_csv[:, 0])[::-1]] # Sort by first column = name of image
        shuffle(bee_csv) # Shuffle data
        # Load predictor data (images)
        bee_imgs = np.empty(len(bee_csv), dtype=object)
        for i in range(len(bee_csv)):
            bee_imgs[i] = imageio.imread(img_path+"/"+bee_csv[i,0]) # First item in bees is image name
        return bee_csv, bee_imgs

    def get_classes(self, bee_targets):
        # Get unique types of species
        self.classes = []
        for item in bee_targets:
            if not item in self.classes:
                self.classes.append(item)
        print("Data classes found:", self.classes)

    def split(self, bee_imgs, bee_targets):
        # Split data into 60% training, 20% validation, and 20% testing
        print("Splitting data...")
        self.training_data = bee_imgs[:int(len(bee_imgs)*.6)]
        self.validation_data = bee_imgs[int(len(bee_imgs)*.6):int(len(bee_imgs)*.8)]
        self.testing_data = bee_imgs[int(len(bee_imgs)*.8):]
        self.training_targets = bee_targets[:int(len(bee_targets)*.6)]
        self.validation_targets = bee_targets[int(len(bee_targets)*.6):int(len(bee_targets)*.8)]
        self.testing_targets = bee_targets[int(len(bee_targets)*.8):]

    def print(self):
        print("Printing info...")
        print("training_data: " + str(len(self.training_data)) + ", training_targets: " + str(len(self.training_targets)))
        print("validation_data: " + str(len(self.validation_data)) + ", validation_targets: " + str(len(self.validation_targets)))
        print("testing_data: " + str(len(self.testing_data)) + ", testing_targets: " + str(len(self.testing_targets)))
