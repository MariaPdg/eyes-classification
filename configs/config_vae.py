""" ______________________________Training parameters________________________________________"""

beta = 0.001
latent_size = 50
batch_size = 64
num_workers = 2
learning_rate = 1e-2
weight_decay = 2e-5
num_iters = 10000
device = 'cuda:0'
pretrained_vae = None

"""_____Data configs (root is specified as a hidden parameter, e.g. project_root/EyeData/...)____"""

# folder containing targets.json and EyesDataset.zip
data_dir = 'EyeData/'
# to save results of training
output_dir = 'output/'
# to save logs
logs_dir = 'logs/'
