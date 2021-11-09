""" ______________________________Training parameters________________________________________"""

latent_size = 50
batch_size = 64
num_workers = 2
learning_rate = 1e-2
weight_decay = 2e-5
num_iters = 10000
device = 'cuda:0'
cls_thresh = 0.5
train_size = 90
valid_size = 10
margin = 1.0

from_scratch = False  # bool, True if training without pre-trained encoder
pretrained_vae = 'vae_20211106-152325'

# [model, iteration] or None if training without pre-trained classifier
# pretrained_cls = ['cls_20211108-202409', 6200]  # eer=0.04
pretrained_cls = None
evaluate = False  # bool, True if only evaluate the model


"""____________________________Paths for inference_______________________________________"""

model_path = 'eyes-classification/output/cls/cls_20211108-202409/cls_6200_20211108-202409.pth'
abs_image_path = 'absolute/path/to/image/image.jpg'

