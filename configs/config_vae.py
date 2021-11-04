import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='path to dataset', type=str)
parser.add_argument('--output', '-o', help='path to save training files', type=str)
parser.add_argument('--logs', '-l', help='path to save logs', type=str)
parser.add_argument('--beta', default=0.001, help='scale for KL divergence in VAE', type=float)
parser.add_argument('--latent_size', default=50, help='dimension of the latent space', type=int)
parser.add_argument('--batch_size', default=128, help='batch size for training', type=int)
parser.add_argument('--num_workers', default=4, help='number of workers for dataloader', type=int)
parser.add_argument('--learning_rate', default=1e-2, help='learning rate for optimizer', type=float)
parser.add_argument('--num_epochs', default=10000, help='number of epochs for training', type=int)
parser.add_argument('--device', default='cuda:0', help='device to use', type=str)
parser.add_argument('--pretrained_vae', '-pretrain_vae', default=None, help='pretrained VAE model', type=str)
parser.add_argument('--pretrained_cls', '-pretrain_cls', default=None, help='pretrained  Classifier', type=str)
parser.add_argument('--cls_threshold', '-cls_thresh', default=0.5, help='threshold for sigmoid output', type=float)

args = parser.parse_args()