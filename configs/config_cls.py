beta = 0.001
latent_size = 50
batch_size = 128
num_workers = 2
learning_rate = 1e-2
weight_decay = 2e-5
num_iters = 5000
device = 'cuda:0'
pretrained_vae = 'vae_20211105-190244'
cls_thresh = 0.5
from_scratch = False
train_size = 40
valid_size = 40
margin = 1.0

# Parameters for inference
pretrained_cls = 'output/cls/cls_20211105-220955/cls_1200_20211105-220955.pth'
abs_image_path = '...'