#!/usr/bin/env python3

from data_lib import *
import matplotlib.pyplot as plt

def main():
    csv_path = "../Data/bee_data.csv"
    img_path = "../Data/bee_imgs"

    bee_csv, bee_imgs = load_data(csv_path, img_path)

    print(bee_csv.shape)
    print(bee_imgs.shape)
    print(bee_imgs[0].shape)

    bee_final_imgs = pad_data(bee_imgs)

    plt.imshow(bee_final_imgs[0])
    plt.show()

main()
