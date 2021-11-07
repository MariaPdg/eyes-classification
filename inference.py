import os
import time
import torch
import logging
import json
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import configs.config_cls as cfg_cls
from utils.models import EyeClassifier


class OpenEyesClassificator(nn.Module):

    """ Class for predictions """

    def __init__(self, pretrained_cls, latent_size=50):
        """
        :param pretrained_cls: string
            Absolute path to pre-trained classifier model
        :param latent_size: int
            Dimension of the latent space
        """
        super().__init__()

        self.model = EyeClassifier(latent_size=latent_size)
        self.model.load_state_dict(torch.load(pretrained_cls), strict=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.train(False)

    def predict(self, inpIm):
        """
        Predicts class of the image
        :param inpIm: string
            Absolute path to the image
        :return: float
            Predicted score between 0 and 1
        """

        img = Image.open(inpIm)
        img_np = 1 / 255 * np.expand_dims(img, 0)
        img_np = np.expand_dims(img_np, 0)
        img_tensor = torch.FloatTensor(img_np)
        pred = self.model(img_tensor)

        return pred.item()

    def visualize(self, inpIm, cls_threshold=0.5, path=None):
        """
        Plots the image with the prediction
        :param inpIm: string
            Absolute path to image
        :param cls_threshold:
            Threshold for the classifier
        :param path: string, optional
            Path to save image
        :return:
        """

        plt.figure()
        img = Image.open(inpIm)
        plt.imshow(img, cmap='gray')
        pred = self.predict(inpIm)

        def to_text(flag):
            return "OPEN" if flag else "CLOSED"

        hard_pred_batch = pred > cls_threshold

        plt.title("pred = {}, score = {:.3f}".format(to_text(hard_pred_batch), pred))

        if path is not None:
            os.makedirs(path, exist_ok=True)
            plt.savefig(path + '/prediction.png')
        else:
            plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', help='root dir for the project', type=str)
    parser.add_argument('--output_dir', '-o', default=cfg_cls.output_dir, help='path to save training files', type=str)
    parser.add_argument('--latent_size', default=cfg_cls.latent_size, help='dimension of the latent space', type=int)
    parser.add_argument('--device', default=cfg_cls.device, help='device to use', type=str)
    parser.add_argument('--abs_model_path', '-cls_path', default=cfg_cls.abs_model_path,
                        help='pretrained  classifier', type=str)
    parser.add_argument('--cls_threshold', '-cls_thresh', default=cfg_cls.cls_thresh,
                        help='threshold for sigmoid output', type=float)
    parser.add_argument('--abs_image_path', '-im_path', default=cfg_cls.abs_image_path, help='absolute image path', type=str)
    parser.add_argument('--message', '-m', default='default message', help='comment to training', type=str)

    args = parser.parse_args()

    # Check available gpu
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestep = time.strftime("%Y%m%d-%H%M%S")

    # Define logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.root, 'logs', timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info("Used device: %s" % device)

    # Create directory to save outputs
    saving_dir = os.path.join(args.root, args.output_dir, 'inference', 'inference_{}'.format(timestep))
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # Save arguments
    with open(os.path.join(saving_dir, 'config_inf.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    model = OpenEyesClassificator(args.abs_model_path, latent_size=args.latent_size)
    pred = model.predict(args.abs_image_path)
    logger.info("Prediction: %f" % pred)
    model.visualize(args.abs_image_path, cls_threshold=args.cls_threshold, path=saving_dir)
