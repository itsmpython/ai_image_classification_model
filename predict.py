#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/predict.py
#                                                                             
# PROGRAMMER: Mallikarjun Somanakatti
# DATE CREATED: 07/10/2018                                  
# REVISED DATE: 07/14/2018 
# PURPOSE:  This program predict the class for an input image by useing a pre-trained model
#           Model is trained using train.py which uses vgg16 model with its classifier reprogrammed.
#
# Use argparse Expected Call with <> indicating expected user input:
# Import Libraries first
import argparse

import matplotlib.pyplot as plt 
import torch 
import torch.nn.functional as F 
from time import time, sleep
from PIL import Image
import numpy as np
import pandas as pd
from torch import nn 
from torch import optim 
from torchvision import datasets, transforms, models 
import helper

def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    
    # Load and rebuild a model
    mymodel = load_checkpoint(in_arg.cp_path, in_arg.checkpoint)
    
    # Process Image before feeding into the model
    input_image = process_image(in_arg.image_path)
    
    # The script successfully reads in an image and a checkpoint 
    # then prints the most likely image class and it's associated probability
    predict(in_arg.image_path, mymodel, in_arg.topK)

# Functions defined below   
def get_input_args():
    # Parsing the arguments
    parser = argparse.ArgumentParser(description='Predicts the class for an input image')
    parser.add_argument('-i', '--image_path', metavar='', help ='Image file path')
    parser.add_argument('-p', '--cp_path', metavar='', help ='Checkpoint path')
    parser.add_argument('-c', '--checkpoint', metavar='', help ='Checkpoint Name')
    parser.add_argument('-K', '--topK', type = int,  metavar = '', help ='List top K classes predicted')
        
    return parser.parse_args()

def load_checkpoint(filepath, checkpoint_name):
    # Loads a checkpoint and rebuilds the model
    checkpoint = torch.load(filepath + checkpoint_name) # load the .pth file
    mymodel = checkpoint['model'] # get the model part from the checkpoint
    mymodel.load_state_dict(checkpoint['state_dict']) #get the state_dict
    
    #print('VGG16 Model Classifier :--> \n', mymodel.classifier)
    #print('Model class_to_idx :--> \n', mymodel.class_to_idx)
    print('Model Checkpoint Loaded')
    return mymodel

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array.
    This is necessary becasue the image used for predcition should normalized similar to the ones used 
    for training the model
    '''
    im = Image.open(image_path)
        
    #RESIZE IMAGE
    im.resize((256, 256))    
    #CROP IMAGE
    width, height = im.size   # Get dimensions
    new_width = 224
    new_height = 224

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    im=im.crop((left, top, right, bottom))
    print('Image Size is : ', im.size)    
    
    #Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1
    # Conver to Numpy array
    np_image = np.array(im)
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    #NORMALIZE
    im = (np_image/255 - means)/stds
    
    # TRANSPOSE IMAGE
    # And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using ndarray.transpose. The color channel needs to be first and retain the order of the other two dimensions.
    img_tp = im.transpose(2, 0, 1)   
    #print('Printing Processed Image :--> \n', img_tp)
    print('Input Image processing completed')
    return img_tp

def predict(image_path, model, topk):
    # Code to predict the class from an image file
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    img = process_image(image_path)    
    
    #Load the model
    #model = load_checkpoint(model)
    model.requires_grad = False
    model.eval()

    # transfer image to tensor
    np_image = np.array(img)
    imgT= torch.from_numpy(np_image)    #From Numpy to torch Tensor

    is_cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_cuda:
        imgT = imgT.cuda()
        model.cuda()
    
        imgS = imgT.unsqueeze_(0).float()
    
        # Run image through model    
        output = model.forward(imgS)  
        ps = torch.exp(output)        
        
        probs, indices = ps.topk(topk)        
        
        # Inorder to access the array from indices (which is a tensor)
        # use .numpy(). We use .cpu() to run it through cpu
        indices = indices.cpu().numpy()[0]
                
        # invert class_to_idx dict        
        idx_to_class = {i:c for c,i in model.class_to_idx.items()}
        classes = {idx_to_class[i] for i in indices}
        
        # Load the json file with flowernames and its class keys
        import json        
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)

        # Iterate through the classes returned by the model and find its corresponding flower_name
        flower_name = []
        for id in classes:
            flower_name.append(cat_to_name[id].capitalize())
            
        # Create a Pandas Series to be used with plotting
        # Create DataFrame from the output data to be used in plotting
        prob_predictions = pd.Series(data=probs.tolist(), index=probs.tolist())
        
        data = {'Flower_Names' : pd.Series(data=flower_name),
                'Probabilities': pd.Series(data=probs.data[0])
               }
        
        df = pd.DataFrame(data)
        print(df);
    
    return probs, classes, flower_name, prob_predictions, df
if __name__ == '__main__':    
    main()   