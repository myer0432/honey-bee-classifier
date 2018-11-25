#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import imageio
from random import shuffle

class data:
    def __init__(self, csv_path, target_feature, img_path):
        # Load data
        bee_csv, bee_imgs = self.load_data(csv_path, img_path)
        bee_imgs = self.pad_data(bee_imgs)
        # Scale image values to a range of 0 to 1
        bee_imgs = bee_imgs / 255.0
        # plt.imshow(bee_final_imgs[0])
        # plt.show()
        # Get just species targets
        bee_targets = bee_csv[:,target_feature]
        # Get classes
        self.get_classes(bee_targets)
        # for target in bee_targets:
        #     target = self.classes.index(target)
        # Split data
        self.split(bee_imgs, bee_targets)

    def load_data(self, csv_path, img_path):
        print("Loading data...")
        bee_csv = pd.read_csv(csv_path) # Read data csv
        bee_csv = bee_csv.values # Convert to numpy array
        bee_csv = bee_csv[np.argsort(bee_csv[:, 0])[::-1]] # Sort by first column = name of image
        filenames = os.listdir(img_path) # Get image paths
        bee_data = np.empty((len(filenames)), dtype=np.object)
        for i in range(len(filenames)):
            bee_data[i] = np.append(bee_csv[i], imageio.imread(img_path+"/"+filenames[i]))
        shuffle(bee_data)
        return bee_data, bee_data[:,-1]

    def pad_data(self, bee_imgs):
        print("Padding...")
        largest_first_dim = 0
        largest_second_dim = 0
        for i in range(bee_imgs.shape[0]):
            if bee_imgs[i].shape[0] > largest_first_dim:
                largest_first_dim = bee_imgs[i].shape[0]
            if bee_imgs[i].shape[1] > largest_second_dim:
                largest_second_dim = bee_imgs[i].shape[1]
        for i in range(bee_imgs.shape[0]):
            first_dim_tuple = (0, largest_first_dim-bee_imgs[i].shape[0])
            second_dim_tuple = (0, largest_second_dim-bee_imgs[i].shape[1])
            third_dim_tuple = (0,0)
            bee_imgs[i] = np.pad(bee_imgs[i], (first_dim_tuple, second_dim_tuple, \
    third_dim_tuple), 'constant', constant_values=0)
        return bee_imgs

    def get_classes(self, bee_targets):
        self.classes = []
        for item in bee_targets:
            if not item in self.classes:
                self.classes.append(item)

    def split(self, bee_imgs, bee_targets):
        print("Splitting data...")
        # Split data into 64% training, 16% validation, and 20% testing
        data = bee_imgs[:int(len(bee_imgs)*.8)]
        self.training_data = data[:int(len(data)*.8)]
        self.validation_data = data[int(len(data)*.8):]
        self.testing_data = bee_imgs[int(len(bee_imgs)*.8):]
        targets = bee_targets[:int(len(bee_targets)*.8)]
        self.training_targets = targets[:int(len(targets)*.8)]
        self.validation_targets = targets[int(len(targets)*.8):]
        self.testing_targets = bee_targets[int(len(bee_targets)*.8):]

    def print(self):
        print("Printing info...")
        print("training_data: " + str(len(self.training_data)) + ", training_targets: " + str(len(self.training_targets)))
        print("validation_data: " + str(len(self.validation_data)) + ", validation_targets: " + str(len(self.validation_targets)))
        print("testing_data: " + str(len(self.testing_data)) + ", testing_targets: " + str(len(self.testing_targets)))
