""" ______________________________Training parameters________________________________________"""

latent_size = 50
batch_size = 64
num_workers = 2
learning_rate = 1e-3
weight_decay = 2e-5
num_iters = 5000
device = 'cuda:0'
cls_thresh = 0.5
train_size = 40
valid_size = 40
margin = 1.0

# bool, True if training without pre-trained encoder
from_scratch = False
pretrained_vae = 'vae_20211106-152325'

# bool, True if only evaluate the model
evaluate = False

# [model, iteration] or None if training without pre-trained classifier
pretrained_cls = None
# pretrained_cls = ['cls_20211106-210609', 1500]
# pretrained_cls = ['cls_20211109-185633', 1600]


"""____________________________Paths for inference_______________________________________"""

model_path = 'eyes-classification/output/cls/cls_20211106-210609/cls_1500_20211106-210609.pth'
abs_image_path = 'images/inf_test2.jpg'  # /absolute/path/to/image/image.jpg

