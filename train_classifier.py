import os
import time
import json
import logging
import argparse
import random

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

import configs.cls_config as cfg_cls
import configs.data_config as cfg_data
from utils.data_loader import load_ann_dataset
from utils.models import EyeClassifier


DEBUG = False


class ClassifierTrainer(object):

    """ A class for classifier training """

    def __init__(self, args, timestep, from_scratch=False, evaluate=False):
        """
        :param args: ArgumentParser object
            Arguments defined in configurations
        :param timestep: string from time.strftime function
            Time value from  to set a unique name
        :param from_scratch: bool, optional
            True if train without pre-trained VAE
        :param evaluate: bool, optional
            True if evaluate model only
        """

        if from_scratch:
            self.pretrained_vae = None
        else:
            self.pretrained_vae = os.path.join(args.root, args.output_dir, 'vae', args.pretrained_vae, args.pretrained_vae + '.pth')
        self.model = EyeClassifier(args.latent_size, self.pretrained_vae)

        self.model.cuda()
        self.cls_threshold = args.cls_threshold
        self.data_dir = os.path.join(args.root, args.data_dir, 'EyesDataset.zip')
        self.labeled_data = os.path.join(args.root, args.data_dir, 'targets.json')
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.pretrained_cls = args.pretrained_cls
        self.num_iters = args.num_iters
        self.train_size = args.train_size
        self.valid_size = args.valid_size
        self.timestep = timestep
        self.evaluate = evaluate
        self.output_dir = os.path.join(args.root, args.output_dir)

        # Create directory to save outputs
        if DEBUG:
            self.saving_dir = os.path.join(args.root, args.output_dir, 'debug', 'cls_{}'.format(self.timestep))
        else:
            self.saving_dir = os.path.join(args.root, args.output_dir, 'cls', 'cls_{}'.format(self.timestep))
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        # Save arguments
        with open(os.path.join(self.saving_dir, 'config_cls.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def train(self):
        """
        Training loop for classifier
        :return:
        """

        writer_train = SummaryWriter(self.saving_dir + '/runs_' + self.timestep + '/train')
        writer_valid = SummaryWriter(self.saving_dir + '/runs_' + self.timestep + '/valid')

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
        logger.info("Len(optimizable_params): %s" % len(optimizable_params))
        optimizer = torch.optim.Adam(optimizable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        # optimizer = torch.optim.RMSprop(optimizable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # Log training and validation metrics
        results = dict(
            iterations=[],
            train_loss=[],
            train_accuracy=[],
            valid_loss=[],
            valid_accuracy=[],
            valid_eer=[],
            threshold=[]
        )

        # Evaluate pre-trained classifier or continue training
        if args.pretrained_cls is not None:
            step_idx = args.pretrained_cls[1]
            model_name = self.pretrained_cls[0].split('_')[0] + '_' + str(args.pretrained_cls[1]) + '_' + \
                         self.pretrained_cls[0].split('_')[1] + '.pth'
            model_dir = os.path.join(self.output_dir, 'cls', self.pretrained_cls[0], model_name)
            if os.path.exists(model_dir):
                logging.info('Load pretrained model {}'.format(model_name))
                self.model.load_state_dict(torch.load(model_dir))
            if self.evaluate:
                valid_loss, valid_accuracy, eer, thresh = self.validate(val_loader, step_idx)
                logging.info(
                    f'Iteration {step_idx}  '
                    f'---- valid loss: {valid_loss.item():.5f} ---- | '
                    f'---- valid accuracy: {valid_accuracy.item():.5f} ----|'
                    f'---- EER: {eer:.5f}---|'
                    f'---- threshold: {thresh:.3f}')
                exit()
        else:
            logging.info('Initialize')

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

                    # Plot some predictions (from training set) and validate training
                    self.plot_predictions(image_batch, pred_batch, target_batch, step_idx)
                    valid_loss, valid_accuracy, eer, thresh = self.validate(val_loader, step_idx)

                    results['iterations'].append(step_idx)
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

                    writer_train.add_scalar('loss', train_loss, step_idx)
                    writer_train.add_scalar('accuracy', train_accuracy, step_idx)
                    writer_valid.add_scalar('loss', valid_loss, step_idx)
                    writer_valid.add_scalar('accuracy', valid_accuracy, step_idx)
                    writer_valid.add_scalar('eer', eer, step_idx)
                    writer_valid.add_scalar('threshold', thresh, step_idx)

                    # Save only if better metrics
                    if len(results['valid_accuracy']) > 2 and results['valid_accuracy'][-1] > results['valid_accuracy'][-2] or \
                            len(results['valid_eer']) > 2 and results['valid_eer'][-1] < results['valid_eer'][-2]:
                        output_dir = os.path.join(self.saving_dir, 'cls_{}_{}.pth'.format(step_idx, timestep))
                        torch.save(self.model.state_dict(), output_dir)

            step_idx += 1

            results_to_save = pd.DataFrame(results)
            output_dir = os.path.join(self.saving_dir, 'cls_{}.csv'.format(timestep))
            results_to_save.to_csv(output_dir, index=False)

            # if len(results['train_loss']) and abs(results['train_loss'][-1] - results['valid_loss'][-1]) > args.margin:
            #     logger.info("Early stopping!")
            #     break

    def plot_predictions(self, image_batch, pred_batch, target_batch, step_idx, split='train'):
        """
        Plots image grid with predictions and targets

        :param image_batch: torch.tensor
            Batch of ground truth images
        :param pred_batch: torch.tensor
            Batch of predicted targets
        :param target_batch: torch.tensor
            Batch of ground truth targets {0, 1}
        :param step_idx:int
            Index of the current iteration
        :param split: string, optional
            Name of data split to visualize, default: 'train'
        :return: plots grid of images with predictions
        """

        def to_text(flag):
            return "OPEN" if flag else "CLOSED"

        hard_pred_batch = pred_batch > self.cls_threshold

        plt.figure()
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        random_list = random.sample(range(0, len(image_batch)), 4)
        idx = 0
        axs = axs.flatten()
        for ax in axs:
            sample_i = random_list[idx]
            ax.imshow(image_batch[sample_i, 0, ...].detach().cpu().numpy(), cmap='gray')
            ax.title.set_text("pred=" + to_text(hard_pred_batch[sample_i]) +
                              " target=" + to_text(target_batch[sample_i]))
            idx += 1

        output_dir = os.path.join(self.saving_dir, 'predictions', split)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_dir + f"/{step_idx:06d}.png")
        plt.close()

    def validate(self, val_loader, step_idx):
        """
        Validation loop for classifier

        :param val_loader: torch.tensor
            Validation dataset
        :param step_idx: int
            Index of the current iteration
        :returns:
            loss: torch.tensor
                Validation loss
            accuracy: torch.tensor
                Validation accuracy
            eer: float
                Validation Equal Error Rate (EER)
            thresh: float
                Threshold corresponding to EER
        """

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

        self.plot_predictions(image_batch, pred_batch, target_batch, step_idx, 'valid')

        fpr, tpr, thresholds = roc_curve(target_batch.cpu(),  pred_batch.cpu(), pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", label="ROC with EER = %0.2f" % eer,)
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.plot([0, 1], [1, 0], linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")

        output_dir = os.path.join(self.saving_dir, 'plots')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_dir + f"/{step_idx:06d}.png")
        plt.close()

        return loss, accuracy, eer, thresh


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', default=cfg_data.root_dir, help='root project directory', type=str)
    parser.add_argument('--data_dir', '-d_dir', default=cfg_data.data_dir, help='dir to dataset', type=str)
    parser.add_argument('--output_dir', '-o', default=cfg_data.output_dir, help='path to save training files', type=str)
    parser.add_argument('--logs_dir', '-l', default=cfg_data.logs_dir, help='path to save logs', type=str)
    parser.add_argument('--latent_size', default=cfg_cls.latent_size, help='dimension of the latent space', type=int)
    parser.add_argument('--batch_size', default=cfg_cls.batch_size, help='batch size for training', type=int)
    parser.add_argument('--num_workers', default=cfg_cls.num_workers, help='number of workers for dataloader', type=int)
    parser.add_argument('--learning_rate', default=cfg_cls.learning_rate, help='learning rate for optimizer', type=float)
    parser.add_argument('--weight_decay', default=cfg_cls.weight_decay, help='weight decay for optimizer', type=float)
    parser.add_argument('--num_iters', default=cfg_cls.num_iters, help='number of iterations for training', type=int)
    parser.add_argument('--margin', default=cfg_cls.margin, help='margin between losses for early stopping', type=float)
    parser.add_argument('--device', default=cfg_cls.device, help='device to use', type=str)
    parser.add_argument('--train_size', '-ts', default=cfg_cls.train_size, help='size of annotated training set', type=int)
    parser.add_argument('--valid_size', '-vs', default=cfg_cls.valid_size, help='size of annotated validation set', type=int)
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

    # Define logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    log_dir = os.path.join(args.root, args.logs_dir)
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info("Used device: %s" % device)

    # Run VAE training
    cls_trainer = ClassifierTrainer(args, timestep, from_scratch=cfg_cls.from_scratch, evaluate=False)
    cls_trainer.train()
