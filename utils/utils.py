import matplotlib.pyplot as plt
import zipfile
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import os
import io
import random


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


if __name__ == "__main__":
    image_list, image_path = load_archive("/home/maria/Study/VisionLabs/EyesDataset.zip")
    print(len(image_list))
    print(image_list[0].shape)
