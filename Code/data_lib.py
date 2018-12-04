#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import imageio
<<<<<<< Updated upstream
from PIL import Image
import matplotlib.pyplot as plt
=======
from sklearn.utils import shuffle
>>>>>>> Stashed changes

def load_data(csv_path, img_path, size_of_set):
    bee_csv = pd.read_csv(csv_path)
    filenames = os.listdir(img_path)
<<<<<<< Updated upstream
    bee_imgs = np.empty((len(filenames)), dtype=np.object)
    for i in range(len(filenames)):
        # print(filenames[i])
        bee_imgs[i] = imageio.imread(img_path+"/"+filenames[i])
=======
    bee_imgs = np.empty(1, dtype=np.object)

    file_count = 0
    object_temp = np.empty(1, dtype=np.object)
    while file_count < len(filenames):
        print(filenames[file_count])

        test_array = imageio.imread(img_path+"/"+filenames[file_count])
        object_temp[0] = test_array

        if test_array.shape[2] == 4:
            bee_csv = bee_csv.drop(file_count, axis=0)
            file_count += 1
        elif file_count == 0:
            bee_imgs[0] = test_array
            file_count += 1
        else:
            bee_imgs = np.concatenate((bee_imgs, object_temp), axis=0)
            file_count += 1

    bee_csv, bee_imgs = shuffle(bee_csv, bee_imgs, random_state=0)
    size_wanted = int(size_of_set*bee_csv.shape[0])
    bee_csv = bee_csv[0:size_wanted]
    bee_imgs = bee_imgs[0:size_wanted]
>>>>>>> Stashed changes

    return bee_csv, bee_imgs

# Sanity check for resizing the images, display the before and after resize
def display_first_image(img_path, resized_img_path):
    filenames = os.listdir(resized_img_path)

    bee_img = imageio.imread(img_path + "/" + filenames[0])
    plt.imshow(bee_img)
    plt.title("Before Processing")
    plt.show()

    bee_img = imageio.imread(resized_img_path + "/" + filenames[0])
    plt.imshow(bee_img)
    plt.title("After Processing")
    plt.show()

# Resizes all images in img_path using a LANCZOS filter and places them
# into resized_img_path
def resize_images(img_path, resized_img_path):
    filenames = os.listdir(img_path)

    print(filenames)

    width, height = find_smallest_dimensions(img_path, filenames)

    for i in np.arange(len(filenames)):

        image = Image.open(img_path + "/" + filenames[i])

        resized_image = image.resize((width, height), Image.LANCZOS)

        resized_image.save(resized_img_path + "/" + filenames[i])

# Find the smallest width and height of the set of files
def find_smallest_dimensions(img_path, filenames):

    # Get size of first image and keep as current smallest dimensions
    image = Image.open(img_path + "/" + filenames[0])
    smallest_width, smallest_height = image.size

    # Loop through rest of files to find minimum width and height
    for i in np.arange(1, len(filenames)):
        image = Image.open(img_path + "/" + filenames[i])
        width, height = image.size

        if (width < smallest_width):
            smallest_width = width

        if (height < smallest_height):
            smallest_height = height

    return width, height

<<<<<<< Updated upstream
# def pad_data(bee_imgs):
#     largest_first_dim = 0
#     largest_second_dim = 0
#
#     for i in range(bee_imgs.shape[0]):
#         if bee_imgs[i].shape[0] > largest_first_dim:
#             largest_first_dim = bee_imgs[i].shape[0]
#
#         if bee_imgs[i].shape[1] > largest_second_dim:
#             largest_second_dim = bee_imgs[i].shape[1]
#
#     # print("Largest first dim:", largest_first_dim)
#     # print("Largest second dim:", largest_second_dim)
#     # print()
#     #
#     # print("Padding...")
#     for i in range(bee_imgs.shape[0]):
#         # print("Image index:", i)
#         # print("\tPre padding size:", bee_imgs[i].shape)
#
#         first_dim_tuple = (0, largest_first_dim-bee_imgs[i].shape[0])
#         second_dim_tuple = (0, largest_second_dim-bee_imgs[i].shape[1])
#         third_dim_tuple = (0,0)
#         bee_imgs[i] = np.pad(bee_imgs[i], (first_dim_tuple, second_dim_tuple, \
# third_dim_tuple), 'constant', constant_values=0)
#     #     print("\tPost padding size:", bee_imgs[i].shape)
#     # print("Done padding")
#     # print()
#
#     return bee_imgs
=======
    return bee_imgs

def split_data(data, labels, ratio):
    bound_1 = int(data.shape[0]*ratio)
    known_samples = data[0:bound_1]
    known_labels = labels[0:bound_1]

    unknown_samples = data[bound_1:data.shape[0]]
    unknown_labels = labels[bound_1:labels.shape[0]]

    return known_samples, known_labels, unknown_samples, unknown_labels
    
>>>>>>> Stashed changes
