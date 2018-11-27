# Flower Image Classifier

Artificially Intelligent, pre-trained program that classifies Images of Flowers from 102 categories listed below with 76% accuracy.
I am using 'vgg16' and 'alexnet' pre-trained networks with customized classifier built on top

Categories of flower images in json format https://github.com/itsmpython/image_classifier/blob/master/cat_to_name.json

** The model uses Pytorch

## Features

1. Image Classifier Project.ipynb is a jupyter notebook that includes implementation details of this project
2. train.py includes program that can train 2 predictive models built of two 
Convolutional neural networks 'Vgg16' and 'Alexnet'. And they have been used with custom classifier built on top
3. predict.py is our prediction program that predicts the top K types of flower images.

## Installation instructions
1. Install the following python libraries
    matplotlib.pyplot  
    torch . 
    torchvision . 
    collections . 
    time . 
    from PIL import Image .
    numpy . 
    pandas . 
    seaborn . 
    json . 
    
2. Run train.py to train the model .
    Example : python train.py -d '~/flowers/' -a 'vgg16' -u 4096 1000 500 -l 0.001 -e 1 -g 'N'

3. Run predict.py to predict the name of an imag . 
    Example: python predict.py -i "~/flowers/test/10/image_07090.jpg" -p "/path/to/imag_classifier/checkpoint" -c "checkpoint.pth" -K 5 -a "vgg16" -g "N"
