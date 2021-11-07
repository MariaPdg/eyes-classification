import os

"""________________________________Data configs ________________________________________________"""

# root directory to the project (nothing to change)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# folder containing targets.json and EyesDataset.zip
data_dir = 'EyeData/'
# to save results of training
output_dir = 'output/'
# to save logs
logs_dir = 'logs/'


