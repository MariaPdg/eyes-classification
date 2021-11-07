# Self-supervised eyes classification

## Motivation¶

Image classification is a well-known problem in pattern recognition and computer vision, which imply assigning an image 
to one of pre-defined classes. Multiple supervised approaches are used in order to solve this problem. In that case, 
a model is trained with image-label pairs. One of the widely used approaches in image processing is convolutional neural
network (CNN), which is able to effectively solve the problem of image classification and recognition. 
However, what happens if the labels are not available during training?  In this case, we have to deal with the problem 
of clustering based on inherited image similarities.

## Problem

In this project we classify open and closed human eyes presented in a dataset without labels. 
Thus, we solve binary classification problem with unlabelled data. The goals of this project:

1. Research the problem of limited size of annotated dataset.
2. Implement an approach based on VAE to classify images with eyes where the final score [0,1], i.e. 0.0 - closed, 1.0 - open.
3. Examine the clustering possibilities of VAE in the latent space

## Approach

The approach consists of two stages. 

* **Stage I:**  We train VAE in an unsupervised manner, i.e. VAE accepts an image as an input and tries to reconstruct this image minimizing loss function. 


* **Stage II:**  we use the pre-trained VAE for the supervised training. For this purpose, we manually annotated 100 samples from the dataset. We freeze the encoder and add one neuron with sigmoid activation function to enable binary classification. 
We set cls_threshold = 0.5. Thus, if the predicted value < 0.5, then the predicted label is "CLOSED", otherwise, "OPEN". 

The full description of this project with results is available in [Notebook](Notebook.ipynb). Some prediction examples:
```html
<figure>
<div align="center">
    <img src=/images/predictions.png" alt="img1" width="800"/>
    <figcaption>Figure 1:  Images with predictions </figcaption>
</div>
</figure>
```
Here you can find a guide how to run the project. 

## Setup

1. Create conda environment

2. Set up project directories: 
```
    project_root
    |          data_dir
    |         |     EyesData.zip
    |         |     targets.json
    + -------- output_dir
    |         |        ...
    + -------- logs_dir
    |         |        ...
    + -------- eyes-classification
``` 
Add corresponding paths to configs:

```python

# folder containing targets.json and EyesDataset.zip
data_dir = 'EyeData/'
# to save results of training
output_dir = 'output/'
# to save logs
logs_dir = 'logs/'

```
Root path will be used as a hidden parameter

## Training

### Stage I: VAE

1. Set up training parameters in [config_vae](configs/config_vae.py)
2. Run [train_vae.py](train_vae.py) with the path to project as a hidden parameter:
```python
python3 train_vae.py -r /path/to/project/root/
```
3. The results will be saved in the directory: `/output_dir/vae/vae_[timestep]`

### Stage II: Classifier

1. Set up training parameters in [config_cls](configs/config_cls.py)
2. Specify path to the pre-trained vae, e.g.
```python
pretrained_vae = 'vae_20211106-152325'
```
3. Run [tran_classifier.py](train_classifier.py) with the path to project as a hidden parameter:

```python
python3 train_classifier.py -r /path/to/project/root/
```

4. The results will be saved in the directory: `/output_dir/cls/cls_[timestep]`

## Inference

1. Specify paths for inference in [config_cls](configs/config_cls.py):

```python
"""___________Parameters for inference_________"""

abs_model_path = 'absolute/path/to/model/model.pth'
abs_image_path = 'absolute/path/to/image/image.jpg'

```
2. Run [inference.py](inference.py) with the path to project as a hidden parameter:

```python
python3 inference.py -r /path/to/project/root/
```
3. The program prints the score from 0 to 1 and  plots the image with a prediction. 