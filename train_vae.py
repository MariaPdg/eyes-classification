import os
import time
import torch
import logging
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from configs.config_vae import args
from utils.utils import load_archive
from utils.models import VAE, kl_divergence


class VaeTrainer(object):

    def __init__(self, args, timestep, pretrain=False):
        self.vae = VAE(latent_size=args.latent_size)
        self.vae.cuda()
        self.data_dir = os.path.join(args.input, 'EyesDataset.zip')
        self.logs_dir = args.logs
        self.saving_dir = os.path.join(args.output, 'vae_{}'.format(timestep))
        self.beta = args.beta
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.timestep = timestep
        self.pretrained_vae = args.pretrained_vae
        self.pretrain = pretrain

    def train(self):

        writer_reconstruction = SummaryWriter(saving_dir + '/runs_' + timestep + '/reconstruction_loss')
        writer_kl = SummaryWriter(saving_dir + '/runs_' + timestep + '/kl_loss')
        writer_vae = SummaryWriter(saving_dir + '/runs_' + timestep + '/vae_loss')

        self.vae.train()

        image_list, image_path = load_archive(self.data_dir)
        tensor_data = (1 / 255. * torch.tensor(image_list).float()).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                             shuffle=True, num_workers=self.num_workers)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=args.learning_rate)

        # results = dict(
        #     epochs=[],
        #     loss_reconstruction=[],
        #     loss_kl=[],
        #     loss_vae=[]
        # )

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
        for idx_epoch in range(self.num_epochs):
            for idx_batch, (batch,) in enumerate(train_loader):
                batch = batch.cuda()
                pred_batch, z_batch, mu_batch, std_batch = self.vae(batch)
                optimizer.zero_grad()
                loss_recon = F.mse_loss(batch, pred_batch)
                loss_kl = self.beta * kl_divergence(mu_batch, std_batch)  # beta-VAE
                loss = loss_recon + loss_kl
                loss.backward()
                optimizer.step()

                if step_idx % 200 == 0:
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(batch[0, 0, ...].detach().cpu().numpy(), cmap='gray')
                    axs[1].imshow(pred_batch[0, 0, ...].detach().cpu().numpy(), cmap='gray')
                    output_dir = os.path.join(self.saving_dir, 'reconstructions')
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(output_dir + f"/{step_idx:06d}.png")
                    # plt.show()
                    plt.close()
                    output_dir = os.path.join(self.saving_dir, 'latent_space')
                    os.makedirs(output_dir, exist_ok=True)
                    plt.figure(figsize=(10, 6))
                    plt.scatter(z_batch.cpu().detach()[..., 0], z_batch.cpu().detach()[..., 1], cmap='rainbow')
                    plt.colorbar()
                    plt.savefig(output_dir + f"/{step_idx:06d}.png")
                    plt.close()

            output_dir = os.path.join(self.saving_dir, 'vae_{}.pth'.format(timestep))
            torch.save(self.vae.state_dict(), output_dir)
            step_idx += 1

            logging.info(
                f'Epoch {idx_epoch}  '
                f'---- reconstruction loss: {loss_recon.item():.5f} ---- | '
                f'---- KLD loss: {loss_kl.item():.5f} ---- | '
                f'---- VAE loss: {loss.item():.5f}')

            writer_reconstruction.add_scalar('reconstruction_loss', loss_recon.item(), idx_epoch)
            writer_kl.add_scalar('kl_loss', loss_kl.item(), idx_epoch)
            writer_vae.add_scalar('vae_loss', loss, idx_epoch)

            # results['epochs'].append(idx_epoch)
            # results['loss_reconstruction'].append(loss_recon.item().detach().cpu().numpy())
            # results['loss_kl'].append(loss_kl.item().detach().cpu().numpy())
            # results['loss_vae'].append(loss.cpu().detach().numpy())
            #
            # results_to_save = pd.DataFrame(results)
            # results_to_save.to_csv(output_dir.replace(".pth", ".csv"), index=False)


if __name__ == "__main__":

    # Check available gpu
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestep = time.strftime("%Y%m%d-%H%M%S")

    # Define logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.logs, timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info("Used device: %s" % device)

    # Create directory to save outputs
    saving_dir = os.path.join(args.output, 'vae_{}'.format(timestep))
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # Save arguments
    with open(os.path.join(saving_dir, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Run VAE training
    vae_trainer = VaeTrainer(args, timestep, pretrain=False)
    vae_trainer.train()