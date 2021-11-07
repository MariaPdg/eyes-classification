""" ______________________________Training parameters________________________________________"""

latent_size = 50
batch_size = 128
num_workers = 2
learning_rate = 1e-3
weight_decay = 2e-5
num_iters = 5000
device = 'cuda:0'
cls_thresh = 0.5
from_scratch = False  # True if training without pre-trained vae
train_size = 40
valid_size = 40
margin = 1.0

# pretrained_vae = 'vae_20211105-190244' eer=0.05
pretrained_vae = 'vae_20211106-152325'
# pretrained_cls = ['cls_20211105-220955', 1200]  # eer=0.05
# pretrained_cls = ['cls_20211106-210609', 1500]  # model, iteration
pretrained_cls = None

"""_____________________________Paths for inference_________________________________________"""

# abs_model_path = '/home/maria/Study/VisionLabs/output/cls/cls_20211105-220955/cls_1200_20211105-220955.pth'
abs_model_path = '/home/maria/Study/VisionLabs/output/cls/cls_20211106-210609/cls_1500_20211106-210609.pth'
abs_image_path = '/home/maria/Study/VisionLabs/EyeData/dataset_B_Eye_Images/closedRightEyes/closed_eye_0116.jpg_face_1_R.jpg'

