#!/usr/bin/env python3

from data_lib import *
import matplotlib.pyplot as plt
import imageio

def main():
    csv_path = "../Data/bee_data.csv"
    img_path = "../Data/bee_imgs"
    resized_img_path = "../Data/resized_bee_imgs"

    # This resizes the images using a LANCZOS filter and places them in the
    # resized_bee_imgs data directory
    resize_images(img_path, resized_img_path)

    # This uses Will's code to load all the images up
    bee_csv, bee_imgs = load_data(csv_path, resized_img_path)

    # Sanity check to look at the first image before and after resizings
    display_first_image(img_path, resized_img_path)

main()
