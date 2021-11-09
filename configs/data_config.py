"""________________________________Data configs ________________________________________________"""

import os

# root directory to the project (nothing to change)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# folder containing targets.json and EyesDataset.zip
data_dir = 'eyes-classification/dataset/'
# to save results of training
output_dir = 'eyes-classification/output/'
# to save logs
logs_dir = 'eyes-classification/logs/'


