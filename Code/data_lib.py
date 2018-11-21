#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import imageio

def load_data(csv_path, img_path):
    bee_csv = pd.read_csv(csv_path)
    filenames = os.listdir(img_path)
    bee_imgs = np.empty((len(filenames)), dtype=np.object)
    for i in range(len(filenames)):
        print(filenames[i])
        bee_imgs[i] = imageio.imread(img_path+"/"+filenames[i])

    print()
    return bee_csv, bee_imgs

def pad_data(bee_imgs):
    largest_first_dim = 0
    largest_second_dim = 0

    for i in range(bee_imgs.shape[0]):
        if bee_imgs[i].shape[0] > largest_first_dim:
            largest_first_dim = bee_imgs[i].shape[0]

        if bee_imgs[i].shape[1] > largest_second_dim:
            largest_second_dim = bee_imgs[i].shape[1]

    print("Largest first dim:", largest_first_dim)
    print("Largest second dim:", largest_second_dim)
    print()

    print("Padding...")
    for i in range(bee_imgs.shape[0]):
        print("Image index:", i)
        print("\tPre padding size:", bee_imgs[i].shape)

        first_dim_tuple = (0, largest_first_dim-bee_imgs[i].shape[0])
        second_dim_tuple = (0, largest_second_dim-bee_imgs[i].shape[1])
        third_dim_tuple = (0,0)
        bee_imgs[i] = np.pad(bee_imgs[i], (first_dim_tuple, second_dim_tuple, \
third_dim_tuple), 'constant', constant_values=0)
        print("\tPost padding size:", bee_imgs[i].shape)
    print("Done padding")
    print()

    return bee_imgs
