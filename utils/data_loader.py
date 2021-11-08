import os
import io
import sys
import json
import zipfile

import torch
import numpy as np
from PIL import Image


def load_archive(ann_data_dir, unlabeled_zip, exclude_ann=False):
    """
    Loads data from zip archive

    :param ann_data_dir: string
        Directory to JSON file with annotated data
    :param unlabeled_zip: string
        Directory to ZIP archive with the whole dataset
    :param exclude_ann: bool, optional
        True if exclude annotated data from the dataset
    :return:
        image_list: list of arrays
            List if images in the array format
        filenames: list of strings
            List of paths to images
    """
    archive = zipfile.ZipFile(unlabeled_zip, "r")
    with open(ann_data_dir, "r") as f:
        ann_dataset = json.load(f)
    num_images = 0
    image_list = []
    filenames = []
    for name in archive.filelist:
        im_id = name.filename.split('/')[1]
        if exclude_ann and (im_id in ann_dataset['open'] or im_id in ann_dataset['closed']):
            continue
        if "__MACOSX" in name.filename:
            continue
        image_bytes = archive.read(name)
        if len(image_bytes) == 0:
            continue
        image = np.array(Image.open(io.BytesIO(image_bytes)))
        image_list.append(image)
        filenames.append(name.filename)
        num_images += 1
    return image_list, filenames


def load_ann_dataset(ann_data_dir, unlabeled_zip, is_train, size=50):
    """
    Loads annotated data from .json file

    :param ann_data_dir: string
        Directory to JSON file with annotated data
    :param unlabeled_zip: string
        Directory to ZIP archive with the whole dataset
    :param is_train: bool
        True if training set
    :param size: int
        Size of the dataset
    :return: dataset: torch.tensor
        Image tensors and corresponding targets
    """
    with open(ann_data_dir, "r") as f:
        dataset = json.load(f)
    flat_list = []
    for label, lst in dataset.items():
        if is_train:
            lst = lst[-size // 2:]
        else:
            lst = lst[:size // 2]
        if label == "open":
            for name in lst:
                # extract unzipped folder name from unlabeled_zip path
                name_folder = unlabeled_zip.split('/')[-1].split('.zip')[0] + '/'
                flat_list.append((name_folder + name, 1))
        else:
            for name in lst:
                flat_list.append((name_folder + name, 0))
    image_list, filenames = load_archive(ann_data_dir, unlabeled_zip)
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

    return images_np, targets_np


class EyesAnnotatedDataloader(object):

    """ Dataloader for annotated data"""

    def __init__(self, ann_data_dir, unlabeled_zip, is_train, size=50, transform=None):
        """

        :param ann_data_dir: string
            Directory to JSON file with annotated data
        :param unlabeled_zip: string
            Directory to ZIP archive with the whole dataset
        :param is_train: bool
            True if training set
        :param size: int
            Size of the dataset
        :param transform: list of transformations
            Transformations applied to images
        """

        self.images, self.targets = load_ann_dataset(ann_data_dir=ann_data_dir, unlabeled_zip=unlabeled_zip,
                         is_train=is_train, size=size)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        image = torch.tensor(self.images[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets


if __name__ == "__main__":

    print(len(sys.argv))
    print(sys.argv)
    data_dir = os.path.join(sys.argv[1], 'EyesDataset.zip')
    image_list, image_path = load_archive(data_dir)
    print(len(image_list))
    print(image_list[0].shape)
