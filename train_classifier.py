import os
import time
import torch
import logging
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import configs.config_cls as cfg_cls
from utils.data_loader import load_ann_dataset
from utils.models import EyeClassifier

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

DEBUG = False

# class EyeDataloader(object):
#
#     def __init__(self, data_json, transform=None):
#
#         with open(data_json, "r") as f:
#             dataset = json.load(f)
#
#     def __len__(self):
#         return len(self.image_names)
#
#     def __getitem__(self, idx):
#
#         image = Image.open(self.image_names[idx])
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image


class ClassifierTrainer(object):

    def __init__(self, args, timestep, from_scratch=False):

        if from_scratch:
            self.pretrained_vae = None
        else:
            self.pretrained_vae = os.path.join(args.output, 'vae', args.pretrained_vae, args.pretrained_vae + '.pth')
        self.model = EyeClassifier(args.latent_size, self.pretrained_vae)

        self.model.cuda()
        self.cls_threshold = args.cls_threshold
        self.data_dir = os.path.join(args.input, 'EyesDataset.zip')
        self.labeled_data = os.path.join(args.input, 'targets.json')
        self.logs_dir = args.logs
        self.saving_dir = os.path.join(args.output, 'cls', 'cls_{}'.format(timestep))
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.timestep = timestep
        self.pretrained_cls = args.pretrained_cls
        self.num_iters = args.num_iters
        self.train_size = args.train_size
        self.valid_size = args.valid_size

        if DEBUG:
            self.saving_dir = os.path.join(args.output, 'debug', 'cls_{}'.format(timestep))
        else:
            self.saving_dir = os.path.join(args.output,  'cls', 'cls_{}'.format(timestep))

    def train(self):

        writer_train = SummaryWriter(saving_dir + '/runs_' + timestep + '/train')
        writer_valid = SummaryWriter(saving_dir + '/runs_' + timestep + '/valid')

        train_dataset = load_ann_dataset(ann_data_dir=self.labeled_data, unlabeled_zip=self.data_dir,
                                         is_train=True, size=self.train_size)
        val_dataset = load_ann_dataset(ann_data_dir=self.labeled_data, unlabeled_zip=self.data_dir,
                                       is_train=False, size=self.valid_size)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset),
                                                   shuffle=True, num_workers=self.num_workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset),
                                                 shuffle=False, num_workers=self.num_workers, drop_last=True)

        all_params = list(self.model.parameters())
        logger.info("Len(parameters): %s" % len(all_params))

        optimizable_params = [p for p in self.model.parameters() if p.requires_grad]
        print("Len(optimizable_params)", len(optimizable_params))
        # optimizer = torch.optim.Adam(optimizable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer = torch.optim.RMSprop(optimizable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        results = dict(
            iteration=[],
            train_loss=[],
            train_accuracy=[],
            valid_loss=[],
            valid_accuracy=[],
            valid_eer=[],
            threshold=[]
        )

        # Load pretrained model
        # if self.pretrained_vae is not None and self.pretrain:
        #     model_dir = os.path.join(args.output, self.pretrained_vae, self.pretrained_vae+'.pth')
        #     if os.path.exists(model_dir):
        #         logging.info('Load pretrained model')
        #         self.vae.load_state_dict(torch.load(model_dir))
        #     else:
        #         logging.info('Initialize')
        # else:
        #     logging.info('Initialize')

        step_idx = 0
        while step_idx < self.num_iters:
            for idx_batch, (image_batch, target_batch) in enumerate(train_loader):
                self.model.train()
                image_batch = image_batch.cuda()
                target_batch = target_batch.cuda()
                pred_batch = self.model(image_batch)
                optimizer.zero_grad()
                train_loss = F.binary_cross_entropy(pred_batch, target_batch)
                train_loss.backward()
                optimizer.step()

                hard_pred_batch = pred_batch > self.cls_threshold
                target_bool_batch = target_batch > self.cls_threshold

                if step_idx % 100 == 0:
                    train_accuracy = torch.sum(torch.eq(hard_pred_batch, target_bool_batch)) / len(target_bool_batch)

                    logging.info(
                        f'Iteration {step_idx}  '
                        f'---- train loss: {train_loss.item():.5f} ---- | '
                        f'---- train accuracy: {train_accuracy.item():.5f}')

                    sample_idx = 0
                    plt.imshow(image_batch[sample_idx, 0, ...].detach().cpu().numpy(), cmap='gray')

                    def to_text(flag):
                        return "OPEN" if flag else "CLOSED"

                    plt.title("pred=" + to_text(hard_pred_batch[sample_idx]) +
                              " target=" + to_text(target_batch[sample_idx]))

                    output_dir = os.path.join(self.saving_dir, 'predictions')
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(output_dir + f"/{step_idx:06d}.png")

                    # plt.show()

                    valid_loss, valid_accuracy, eer, thresh = self.validate(val_loader)
                    results['iteration'].append(step_idx)
                    results['train_loss'].append(train_loss.item())
                    results['train_accuracy'].append(train_accuracy.item())
                    results['valid_loss'].append(valid_loss.item())
                    results['valid_accuracy'].append(valid_accuracy.item())
                    results['valid_eer'].append(eer)
                    results['threshold'].append(thresh.item())

                    logging.info(
                        f'Iteration {step_idx}  '
                        f'---- valid loss: {valid_loss.item():.5f} ---- | '
                        f'---- valid accuracy: {valid_accuracy.item():.5f} ----|'
                        f'---- EER: {eer:.5f}---|'
                        f'---- threshold: {thresh:.3f}')

            if len(results['valid_accuracy']) > 2 and results['valid_accuracy'][-1] > results['valid_accuracy'][-2]:
                output_dir = os.path.join(self.saving_dir, 'cls_{}.pth'.format(timestep))
                torch.save(self.model.state_dict(), output_dir)

            step_idx += 1

            writer_train.add_scalar('loss', train_loss, step_idx)
            writer_valid.add_scalar('loss', valid_loss, step_idx)

            writer_train.add_scalar('accuracy', train_accuracy, step_idx)
            writer_valid.add_scalar('accuracy', valid_accuracy, step_idx)

            results_to_save = pd.DataFrame(results)
            output_dir = os.path.join(self.saving_dir, 'cls_{}.csv'.format(timestep))
            results_to_save.to_csv(output_dir, index=False)

    def validate(self, val_loader):

        self.model.train(False)
        image_batch, target_batch = next(iter(val_loader))
        image_batch = image_batch.cuda()
        target_batch = target_batch.cuda()
        with torch.no_grad():
            pred_batch = self.model(image_batch)
            loss = F.binary_cross_entropy(pred_batch, target_batch)
        hard_pred_batch = pred_batch > self.cls_threshold
        target_bool_batch = target_batch > self.cls_threshold
        accuracy = torch.sum(torch.eq(hard_pred_batch, target_bool_batch)) / len(target_bool_batch)

        fpr, tpr, thresholds = roc_curve(target_batch.cpu(),  pred_batch.cpu(), pos_label=1)

        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)

        return loss, accuracy, eer, thresh


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='path to dataset', type=str)
    parser.add_argument('--output', '-o', help='path to save training files', type=str)
    parser.add_argument('--logs', '-l', help='path to save logs', type=str)
    parser.add_argument('--beta', default=cfg_cls.beta, help='scale for KL divergence in VAE', type=float)
    parser.add_argument('--latent_size', default=cfg_cls.latent_size, help='dimension of the latent space', type=int)
    parser.add_argument('--batch_size', default=cfg_cls.batch_size, help='batch size for training', type=int)
    parser.add_argument('--num_workers', default=cfg_cls.num_workers, help='number of workers for dataloader', type=int)
    parser.add_argument('--learning_rate', default=cfg_cls.learning_rate, help='learning rate for optimizer', type=float)
    parser.add_argument('--weight_decay', default=cfg_cls.weight_decay, help='weight decay for optimizer', type=float)
    parser.add_argument('--num_epochs', default=cfg_cls.num_epochs, help='number of epochs for training', type=int)
    parser.add_argument('--num_iters', default=cfg_cls.num_iters, help='number of iterations for training', type=int)
    parser.add_argument('--device', default=cfg_cls.device, help='device to use', type=str)
    parser.add_argument('--train_size', default=cfg_cls.train_size, help='size of annotated training set', type=float)
    parser.add_argument('--valid_size', default=cfg_cls.valid_size, help='size of annotated validation set', type=float)
    parser.add_argument('--pretrained_vae', '-pretrain_vae', default=cfg_cls.pretrained_vae,
                        help='pretrained VAE model', type=str)
    parser.add_argument('--pretrained_cls', '-pretrain_cls', default=cfg_cls.pretrained_cls,
                        help='pretrained  classifier', type=str)
    parser.add_argument('--cls_threshold', '-cls_thresh', default=cfg_cls.cls_thresh,
                        help='threshold for sigmoid output', type=float)
    parser.add_argument('--message', '-m', default='default message', help='comment to training', type=str)

    args = parser.parse_args()

    # Check available gpu
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestep = time.strftime("%Y%m%d-%H%M%S")

    # Define loggin
    # g
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.logs, timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info("Used device: %s" % device)

    # Create directory to save outputs
    if DEBUG:
        saving_dir = os.path.join(args.output, 'debug', 'cls_{}'.format(timestep))
    else:
        saving_dir = os.path.join(args.output, 'cls', 'cls_{}'.format(timestep))
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # Save arguments
    with open(os.path.join(saving_dir, 'config_cls.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Run VAE training
    cls_trainer = ClassifierTrainer(args, timestep, from_scratch=cfg_cls.from_scratch)
    cls_trainer.train()