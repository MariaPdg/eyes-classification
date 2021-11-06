import os
import time
import json
import logging
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import configs.config_vae as cfg_vae
import utils.models as models
import utils.data_loader as utils

DEBUG = False


class VaeTrainer(object):

    def __init__(self, args, timestep, pretrain=False):

        self.vae = models.VAE(latent_size=args.latent_size)
        self.vae.cuda()
        self.data_dir = os.path.join(args.root, args.data_dir, 'EyesDataset.zip')
        self.labeled_data = os.path.join(args.root, args.data_dir, 'targets.json')
        self.beta = args.beta
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.learning_rate = args.learning_rate
        self.pretrained_vae = args.pretrained_vae
        self.pretrain = pretrain
        self.num_iters = args.num_iters
        self.timestep = timestep

        # Create directory to save outputs
        if DEBUG:
            self.saving_dir = os.path.join(args.root, args.output_dir, 'debug', 'vae_{}'.format(self.timestep))
        else:
            self.saving_dir = os.path.join(args.root, args.output_dir, 'vae', 'vae_{}'.format(self.timestep))
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        # Save arguments
        with open(os.path.join(self.saving_dir, 'config_vae.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def train(self):

        writer = SummaryWriter(self.saving_dir + '/runs_' + self.timestep)

        self.vae.train()

        image_list, image_path = utils.load_archive(self.labeled_data, self.data_dir, exclude_ann=True)
        tensor_data = (1 / 255. * torch.tensor(image_list).float()).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                             shuffle=True, num_workers=self.num_workers)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=args.learning_rate)

        results = dict(
            iterations=[],
            loss_reconstruction=[],
            loss_kl=[],
            loss_vae=[]
        )

        # Load pretrained model
        if self.pretrained_vae is not None and self.pretrain:
            model_dir = os.path.join(args.output, self.pretrained_vae, self.pretrained_vae+'.pth')
            if os.path.exists(model_dir):
                logging.info('Load pretrained model')
                self.vae.load_state_dict(torch.load(model_dir))
            else:
                logging.info('Initialize')
        else:
            logging.info('Initialize')

        step_idx = 0
        while step_idx < self.num_iters:
            for idx_batch, (batch,) in enumerate(train_loader):
                batch = batch.cuda()
                pred_batch, z_batch, mu_batch, std_batch = self.vae(batch)
                optimizer.zero_grad()
                loss_recon = F.mse_loss(batch, pred_batch)
                loss_kl = self.beta * models.kl_divergence(mu_batch, std_batch)  # beta-VAE
                loss = loss_recon + loss_kl
                loss.backward()
                optimizer.step()

                if step_idx % 100 == 0:
                    fig, axs = plt.subplots(2, 2)
                    axs[0, 0].imshow(batch[0, 0, ...].detach().cpu().numpy(), cmap='gray')
                    axs[0, 0].title.set_text('Ground truth')
                    axs[0, 1].imshow(pred_batch[0, 0, ...].detach().cpu().numpy(), cmap='gray')
                    axs[0, 1].title.set_text('Reconstructions')
                    axs[1, 0].imshow(batch[1, 0, ...].detach().cpu().numpy(), cmap='gray')
                    axs[1, 1].imshow(pred_batch[1, 0, ...].detach().cpu().numpy(), cmap='gray')
                    output_dir = os.path.join(self.saving_dir, 'reconstructions')
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(output_dir + f"/{step_idx:06d}.png")
                    plt.close()
                    # visualize the first 2 components of the latent space
                    output_dir = os.path.join(self.saving_dir, 'latent_space')
                    os.makedirs(output_dir, exist_ok=True)
                    plt.figure(figsize=(8, 5))
                    plt.scatter(z_batch.cpu().detach()[..., 0], z_batch.cpu().detach()[..., 1], cmap='rainbow')
                    plt.colorbar()
                    plt.savefig(output_dir + f"/{step_idx:06d}.png")
                    plt.close()

                    writer.add_scalars('loss', {
                        'reconstruction': loss_recon.item(),
                        'kl':  loss_kl.item(),
                    }, step_idx)
                    writer.add_scalar('loss_vae', loss, step_idx)

                    # Log model per epoch
                    output_dir = os.path.join(self.saving_dir, 'vae_{}.pth'.format(timestep))
                    torch.save(self.vae.state_dict(), output_dir)

                    logging.info(
                        f'Iteration {step_idx}  '
                        f'---- reconstruction loss: {loss_recon.item():.5f} ---- | '
                        f'---- KLD loss: {loss_kl.item():.5f} ---- | '
                        f'---- VAE loss: {loss.item():.5f}')

                    results['iterations'].append(step_idx)
                    results['loss_reconstruction'].append(loss_recon.item())
                    results['loss_kl'].append(loss_kl.item())
                    results['loss_vae'].append(loss.item())

                step_idx += 1

            results_to_save = pd.DataFrame(results)
            output_dir = os.path.join(self.saving_dir, 'vae_{}.csv'.format(timestep))
            results_to_save.to_csv(output_dir, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', help='root project directory', type=str)
    parser.add_argument('--data_dir', '-d_dir', default=cfg_vae.data_dir, help='path to the dataset', type=str)
    parser.add_argument('--output_dir', '-o', default=cfg_vae.output_dir, help='path to save training files', type=str)
    parser.add_argument('--logs_dir', '-l', default=cfg_vae.logs_dir, help='path to save logs', type=str)
    parser.add_argument('--beta', '-be', default=cfg_vae.beta, help='scale for KL divergence in VAE', type=float)
    parser.add_argument('--batch_size', '-bs', default=cfg_vae.batch_size, help='batch size for training', type=int)
    parser.add_argument('--num_iters', '-ni', default=cfg_vae.num_iters, help='number of epochs for training', type=int)
    parser.add_argument('--device', '-d', default='cuda:0', help='device to use', type=str)
    parser.add_argument('--message', '-m', default='default message', help='comment to training', type=str)
    parser.add_argument('--pretrained_vae', '-vae', default=cfg_vae.pretrained_vae, help='pretrained VAE model',
                        type=str)
    parser.add_argument('--latent_size', '-ls', default=cfg_vae.latent_size, help='dimension of the latent space',
                        type=int)
    parser.add_argument('--num_workers', '-nw', default=cfg_vae.num_workers, help='number of workers for dataloader',
                        type=int)
    parser.add_argument('--learning_rate', '-lr', default=cfg_vae.learning_rate, help='learning rate for optimizer',
                        type=float)

    args = parser.parse_args()

    # Check available gpu
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestep = time.strftime("%Y%m%d-%H%M%S")

    # Define logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.root, args.logs_dir, timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info("Used device: %s" % device)

    # Run VAE training
    vae_trainer = VaeTrainer(args, timestep, pretrain=False)
    vae_trainer.train()