import os
import sys
import torch
import torch.nn as nn
from collections import OrderedDict


class BatchNorm(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
        self.bn = nn.BatchNorm2d(op.out_channels)

    def forward(self, x):
        x = self.op(x)
        x = self.bn(x)
        return x


class Encoder(nn.Module):

    def __init__(self, latent_size):
        super().__init__()
        self.conv1 = BatchNorm(nn.Conv2d(1, 32, 3, padding=1, stride=2))
        self.conv2 = BatchNorm(nn.Conv2d(32, 64, 3, padding=1, stride=2))
        self.conv3 = BatchNorm(nn.Conv2d(64, 128, 3, padding=1, stride=2))
        self.conv4 = BatchNorm(nn.Conv2d(128, 256, 3, padding=0, stride=3))

        self.mu_fc = nn.Linear(256, latent_size)
        self.log_var_fc = nn.Linear(256, latent_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)

        mu = self.mu_fc(x)
        log_var = self.log_var_fc(x)
        return mu, log_var


class Decoder(nn.Module):

    def __init__(self, latent_size):
        super().__init__()

        self.dec_fc = nn.Linear(latent_size, 256)

        self.tconv1 = nn.ConvTranspose2d(256, 128, 3, padding=0, stride=3)
        self.tconv2 = nn.ConvTranspose2d(128, 64, 3, output_padding=0, stride=2)
        self.tconv3 = nn.ConvTranspose2d(64, 32, 3, output_padding=0, stride=2)
        self.tconv4 = nn.ConvTranspose2d(32, 1, 3, output_padding=0, stride=2)

    def forward(self, z):
        x = self.dec_fc(z)

        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.relu(self.tconv1(x))
        x = torch.relu(self.tconv2(x))[:, :, :-1, :-1]
        x = torch.relu(self.tconv3(x))[:, :, :-1, :-1]
        x = torch.sigmoid(self.tconv4(x))[:, :, :-1, :-1]

        return x


class VAE(nn.Module):

    def __init__(self, latent_size):
        super().__init__()

        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        mu, log_var = self.encoder(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        x = self.decoder(z)

        return x, z, mu, std


def kl_divergence(mu, std):

    kl = (std ** 2 + mu ** 2 - torch.log(std) - 1 / 2).mean()
    return kl


class EyeClassifier(nn.Module):

    def __init__(self, latent_size, pretrained_vae=None):
        super().__init__()
        self.encoder = Encoder(latent_size)
        # self.freeze_backbone = args.pretrained_vae is not None
        if pretrained_vae is not None:
            state_dict = torch.load(pretrained_vae)
            state_dict = OrderedDict(((k[len("encoder."):], v)
                                      for k, v in state_dict.items()
                                      if "encoder." in k))  # while encoder
            self.encoder.load_state_dict(state_dict, strict=True)
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.encoder.eval()
        self.class_fc = nn.Linear(latent_size, 1)

    def forward(self, x):
        # with torch.no_grad() if self.freeze_backbone else contextlib.nullcontext():
        mu, log_var = self.encoder(x)
        logits = self.class_fc(mu)
        x = torch.sigmoid(logits)
        x = x.squeeze(-1)
        return x

    def train(self, mode=True):
        self.encoder.train(False)
        self.class_fc.train(mode)


if __name__ == "__main__":

    print(len(sys.argv))
    print(sys.argv)
    data_dir = os.path.join(sys.argv[1], 'vae_20211104-213016/vae_20211104-213016.pth')
    vae = VAE(latent_size=50).cuda()
    classifier = EyeClassifier(latent_size=50, pretrained_vae=data_dir).cuda()
