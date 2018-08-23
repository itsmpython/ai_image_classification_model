#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/train.py
#                                                                             
# PROGRAMMER: Mallikarjun Somanakatti
# DATE CREATED: 07/06/2018                                  
# REVISED DATE: 07/23/2018 
# PURPOSE: Image Classifier Project for the AIPND Program
#           Check flower images & report results: read them in, predict their
#           class, compare prediction to actual value labels from the cat_to_name.json file
#           and output results
#
# Use argparse Expected Call with <> indicating expected user input:

# Import Lobraries first
import argparse
import matplotlib.pyplot as plt 
import torch 
import torch.nn.functional as F 
from time import time, sleep
from PIL import Image
import numpy as np
from torch import nn 
from torch import optim 
from torchvision import datasets, transforms, models 
import helper 

# Importing Custom Libraries
import load_and_transform_images as lt
import load_pretrained_model as pm

def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    
    # Loads the dataset and transforms images using pytorch transforms + compose functions 
    data = loads_transforms_data(in_arg.data_dir)
    
    # Loads a pre-trained model and updates is classifier to define out model
    model = load_pretrained_model(in_arg.arch, in_arg.hiddenunits)
    
    # Trains the model and prints running and training loss
    trained_model = train_model(model, data[3], data[4], in_arg.learn_rate, in_arg.epochs, in_arg.gpu)
    
    # Tests the model for test images (From test data set) and prints accurcy
    test_network_model(trained_model[0], data[5])
    
    # Save the Model Checkpoint    
    save_model_checkpoint(trained_model[0], in_arg.arch, in_arg.epochs, in_arg.learn_rate, trained_model[2], in_arg.hiddenunits, data[0]) 
    
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
# Functions defined below   
def get_input_args():
    # Parsing the arguments
    parser = argparse.ArgumentParser(description='Trains a Model and saves the Checkpoint')
    parser.add_argument('-d', '--data_dir', metavar='', help ='Image Data Directory to train, validate and test')
    parser.add_argument('-a', '--arch', metavar='', help ='Neural Network Architecture(coded for ''vgg16'' and ''alexnet'' for now)')
    parser.add_argument('-u', '--hiddenunits', metavar='', type = int, nargs='+', help ='Number of hidden Units')
    parser.add_argument('-l', '--learn_rate', type = float, metavar = '', help ='Learning Rate set for training the model')
    parser.add_argument('-e', '--epochs', metavar='', type = int, help ='Number of Epochs')
    parser.add_argument('-g', '--gpu', metavar='', help ='Pass ''Y'' or ''N'' for gpu')
    
    return parser.parse_args()

def loads_transforms_data(data_dir):
    # Reading the data set, transforming the images and then loading the data
    print ('Read, transform and Load the data sets')
    # Invoke a function "transform_image_data" that transforms the images and provides datasets
    transformed_data = lt.transform_image_data(data_dir, 'train/','valid/','test/')
    train_datasets = transformed_data[0]
    valid_datasets = transformed_data[1]
    test_datasets = transformed_data[2]
    trainloader = transformed_data[3]
    validloader = transformed_data[4]
    testloader = transformed_data[5]
    #return data[0], data[1], data[3], data[4]
    return train_datasets, valid_datasets, test_datasets, trainloader, validloader, testloader    

def load_pretrained_model(arch, hiddenunits):
    # Invoke a function "load_oretrained_model" which loads a pretrained vgg16model and
    # replaces its classifier according to the data we like to classify
    mymodel = pm.load_pretrained_model(arch, hiddenunits)
    return mymodel

def train_model(model, train_data, validation_data, learn_rate, nepochs, gpu):
    # Define HYPER Parameters    
    learn_rate = learn_rate
    epochs = nepochs
    training_loss = 0
    running_loss = 0
    test_loss = 0     
    print_every = 40
    steps = 0
    accuracy = 0
    validation_loss = 0
    
    #myvgg16model = load_pretrained_model()    
    modeltotrain = model
    print('train.py modeltotrain.classifier = \n', modeltotrain.classifier)

    # Define Loss Criteria and the Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(modeltotrain.classifier.parameters(), lr = learn_rate)
    print(learn_rate)
    print('Printing Optimizer', optimizer)
    
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    if gpu == 'Y':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device : ", device)
    else:
        device = "cpu"
        print("Device : ", device)

    modeltotrain.to(device)
    #print(vgg16model.to(device))
  
    epochs = nepochs
    steps = steps
    running_loss = running_loss
    print_every = print_every

    for e in range(epochs):
        running_loss = 0
        modeltotrain.train()

        ''' since we are using pre-trained networks we use "inputs," instead of "images."
        As in: for ii, (inputs, labels) in enumerate(trainloader):.
        And therefore do not need to resize an image before forward pass as: outputs = model.forward(inputs)
        Because of this we do not need to use "for images, labels in iter(trainloader)" '''     

        trainloader = train_data
        validloader = validation_data

        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            #Perform Forward and Backward passes
            optimizer.zero_grad() 
            outputs = modeltotrain.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 
            test_loss += criterion(outputs, labels).item()            
            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            
            if steps % print_every == 0:
                modeltotrain.eval()
                
                for ii, (inputs, labels) in enumerate(validloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    predicted_outputs = modeltotrain.forward(inputs)
                    validation_loss = criterion(predicted_outputs, labels).item()
                    # Calculating the accuracy
                    ps = torch.exp(predicted_outputs).data
                    
                    # Class with the highest probability is our predicted class
                    equality = (labels.data == ps.max(dim=1)[1])
                    accuracy = equality.type(torch.FloatTensor).mean()
                        
                print("Epoch : {}/{}...".format(e+1, epochs),
                      "Training Loss :{:4f}".format(running_loss/print_every),
                      "Validation Loss :{:3f}".format(validation_loss),
                      "Validation Accuracy :{:3f}".format(accuracy))
            
                running_loss = 0
                modeltotrain.train()
    return modeltotrain, criterion, optimizer

def test_network_model(model, test_data):
    
    accuracy = 0
    total = 0    
    myvgg16model = model
    testloader = test_data
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for ii, (images, labels) in enumerate(testloader):
            myvgg16model.eval()
            #images, labels = images.to('cpu'), labels.to('cpu')
            images, labels = images.to(device), labels.to(device)

            outputs = myvgg16model(images)
            _, predicted = torch.max(outputs.data, 1)            
            total += labels.size(0)

            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += (predicted == labels).sum().item()            
        print('\n\n Validation Accuracy of the network on the test images: %d %%' % (100 * accuracy / total))    
        return accuracy

def save_model_checkpoint(model, arch, epochs, learn_rate, optimizer, hidden_units, train_datasets):
    
    # Following is ond of the methogs to save without saving other details of the model    
    
    if arch == 'vgg16':
        input_size = 25088
    elif arch == 'alexnet':
        input_size = 9216
        
    hidden_sizes = hidden_units
    output_size = 102
    
    # Saving class_to_index
    #vgg16model.class_to_idx = train_datasets.class_to_idx
    model.class_to_idx = train_datasets.class_to_idx

    ## We probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets
    ## image_datasets['train'].class_to_idx. 
    ## You can attach this to the model as an attribute which makes inference easier later on.        
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_sizes,
                  'batch_size': 32,
                  'learning_rate': learn_rate,
                  'model_name': model,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epochs,
                  'class_to_idx': model.class_to_idx} 
    
    if arch == 'vgg16':
        #torch.save(checkpoint, 'vgg16checkpoint.pth', '/home/workspace/aipnd-project') 
        torch.save(checkpoint, 'checkpoint.pth') 
        print('\n VGG16 Model Checkpoint saved')
    elif arch == 'alexnet':
        torch.save(checkpoint, 'alexnetcheckpoint.pth') 
        print('\n Alexnet Checkpoint saved')
    
if __name__ == '__main__':    
    main()