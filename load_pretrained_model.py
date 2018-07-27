# coding: utf-8
# Imports libraries here 

from torchvision import datasets, transforms, models 
import torch 
from collections import OrderedDict
from torch import nn

def load_pretrained_model(arch, hidden_units):    
    hidden_sizes = hidden_units #[4096, 1000, 500]
    output_size = 102
    
    print('\n Hidden units in the model are', hidden_units)
    if arch == 'vgg16':
        print('\n Loading vgg16 model')
        input_size = 25088
        vgg16model = models.vgg16(pretrained=True)
        #vgg16model 
        #print('Printing vgg16model.classifier \n' ,vgg16model.classifier)
        for params in vgg16model.parameters():
            params.requires_grad = False 
    elif arch == 'alexnet':
        print('\n Loading alexnet model')
        input_size = 9216
        alexnet = models.alexnet(pretrained=True)
        for params in alexnet.parameters():
            params.requires_grad = False

    # 2 ## Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout 
    myclassifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_sizes[0])),
                                              ('relu1', nn.ReLU()),
                                              ('fc1_dropout',nn.Dropout(p=0.3)),
                                              ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                                              ('relu2', nn.ReLU()),
                                              ('fc2_dropout',nn.Dropout(p=0.3)),
                                              ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                                              ('relu3', nn.ReLU()),
                                              ('fc3_dropout',nn.Dropout(p=0.3)),
                                              ('fc4', nn.Linear(hidden_sizes[2], output_size)),
                                              ('output',nn.LogSoftmax(dim=1))
                                              ## Using LogSoftMax because the numbers that are returned as closer to 0 0r 1
                                              ## basically a floating num. Log Soft Max keeps the numbers away from 0 and 1
                                             ]))
            
    if arch == 'vgg16':
        vgg16model.classifier = myclassifier
        #print(vgg16model.classifier)
        print('\n Using ''vgg16'' model')
        model_arch = vgg16model
    
    elif arch == 'alexnet':
        alexnet.classifier = myclassifier
        print('\n Using ''alexnet'' model')
        model_arch = alexnet
        
    return model_arch