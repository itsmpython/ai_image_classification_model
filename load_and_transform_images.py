# coding: utf-8
# Imports libraries here 

'''
Training data augmentation, normalization, batching and loading takes place in this function.
Torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping
'''
from torchvision import datasets, transforms, models 
import torch 
import helper 

def transform_image_data(data_dir, train_dir, valid_dir, test_dir):
    ''' transforms the image file using pytorch transforms and Compose methods'''
    
    data_dir = data_dir
    train_dir = data_dir + train_dir
    valid_dir = data_dir + valid_dir
    test_dir = data_dir + test_dir
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224), 
                                        transforms.RandomHorizontalFlip(), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                                        ])
    valid_transforms = transforms.Compose([transforms.Resize(256), 
                                            transforms.RandomResizedCrop(224), 
                                            transforms.ToTensor()
                                            ]) 

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.RandomResizedCrop(224), 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                                        ]) 

# TODO: Load the datasets with ImageFolder 

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders 

    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)

    #print ('train_datasets, valid_datasets, test_datasets --> ', train_datasets, valid_datasets, test_datasets)
    return train_datasets, valid_datasets, test_datasets, trainloader, validloader, testloader