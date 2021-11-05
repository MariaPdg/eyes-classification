import matplotlib.pyplot as plt
import zipfile
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import os
import io
import random
import json
import torch
import sys


def load_archive(data_dir):
    archive = zipfile.ZipFile(data_dir, "r")
    num_images = 0
    image_list = []
    filenames = []
    for name in archive.filelist:
        if "__MACOSX" in name.filename:
            continue
        image_bytes = archive.read(name)
        if len(image_bytes) == 0:
            continue
        image = np.array(Image.open(io.BytesIO(image_bytes)))
        image_list.append(image)
        filenames.append(name.filename)
        num_images += 1
    #     if False and num_images < 10:
    #         plt.imshow(image, cmap='gray')
    #         plt.show()
    # images = np.array(image_list)
    return image_list, filenames


def load_dataset(labeled_data_dir, unlabeled_zip, is_train):
    with open(labeled_data_dir, "r") as f:
        dataset = json.load(f)
    flat_list = []
    for label, lst in dataset.items():
        val_div = 2
        if is_train:
            lst = lst[len(lst) // val_div:]
        else:
            lst = lst[:len(lst) // val_div]
        if label == "open":
            for name in lst:
                flat_list.append(('EyesDataset/' + name, 1))
        else:
            for name in lst:
                flat_list.append(('EyesDataset/' + name, 0))
    image_list, filenames = load_archive(unlabeled_zip)
    name_to_image = dict(zip(filenames, image_list))
    images = []
    for im_name, im_target in flat_list:
        if im_name in name_to_image:
            images.append(name_to_image[im_name])
        else:
            print("Warning: file not found")
    images_np = np.array(images)
    images_np = 1 / 255 * np.expand_dims(images_np, 1)
    targets_np = np.array([v[1] for v in flat_list])
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(images_np, dtype=torch.float32),
        torch.tensor(targets_np, dtype=torch.float32))
    return dataset


if __name__ == "__main__":

    print(len(sys.argv))
    print(sys.argv)
    data_dir = os.path.join(sys.argv[1], 'EyesDataset.zip')
    image_list, image_path = load_archive(data_dir)
    print(len(image_list))
    print(image_list[0].shape)
