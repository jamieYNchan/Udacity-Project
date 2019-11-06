#!/usr/bin/env python
# coding: utf-8

# In[110]:


import torch
import time
import json
import os
import numpy as np

from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm


# In[198]:


class data_transforms:
    train_transforms = None
    test_transforms = None
    validation_transforms = None
    
class image_datasets:
    train_data , test_data, valid_data = None, None, None
class dataloaders:
    trainloader, testloader, validloader = None, None, None
    


# In[315]:


def init(gpu):
    global device
    if torch.cuda.is_available() and gpu:
        print("Changed to GPU mode.")
        device = torch.device("cuda")
    elif not torch.cuda.is_available():
        print('GPU is not available.')
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'
    
    data_transforms.train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                           transforms.RandomResizedCrop(224),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                                [0.229, 0.224, 0.225])])
    data_transforms.test_transforms = transforms.Compose([transforms.Resize(255),
                                                          transforms.CenterCrop(244),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                               [0.229, 0.224, 0.225])])
    data_transforms.validation_transforms = transforms.Compose([transforms.Resize(255),
                                                               transforms.CenterCrop(244),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                                    [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets.train_data = datasets.ImageFolder(train_dir, transform=data_transforms.train_transforms)
    image_datasets.test_data = datasets.ImageFolder(test_dir, transform=data_transforms.test_transforms)
    image_datasets.valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms.validation_transforms)
    
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders.trainloader = torch.utils.data.DataLoader(image_datasets.train_data, batch_size = 64, shuffle=True)
    dataloaders.testloader = torch.utils.data.DataLoader(image_datasets.test_data, batch_size = 32)
    dataloaders.validloader = torch.utils.data.DataLoader(image_datasets.valid_data, batch_size = 32)
    
    print("Loaded Data")
    return data_transforms, image_datasets, dataloaders


def create_model(structure = 'densenet121', dropout = 0.3, lr = 0.003, hidden_units = 512, checkpoint = None):
    global model
    global criterion
    criterion = nn.NLLLoss()
    global optimizer
    
    if checkpoint is None:
        output_size = 102
    
        if structure == 'densenet121':
            model = models.densenet121(pretrained=True)
            input_size = 1024
        elif structure == 'vgg16':
            model = models.vgg16(pretrained=True)
            input_size = 25088
        elif structure == 'vgg13':
            model = models.vgg13(pretrained=True)
            input_size = 4096
    
        if not model.classifier is None:
            if type(model.classifier) is nn.Linear:
                input_size = model.classifier.in_features
            else:
                input_size = model.classifier[0].in_features
            
        print(f"Input size: {input_size}")
        print("Created model.")
        for param in model.parameters():
            param.requires_grad = False
    
    
        model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                                      ('relu1', nn.ReLU()),
                                                      ('dropout1', nn.Dropout(p = dropout)),
                                                      ('fc2', nn.Linear(hidden_units, output_size)),
                                                      ('output', nn.LogSoftmax(dim = 1))
                                                     ]))
        optimizer = optim.Adam(model.parameters(), lr)
    else:
        structure = checkpoint['structure']
        epochs = checkpoint['epochs']
        dropout = checkpoint['dropout']
        
        if structure == 'densenet121':
            model = models.densenet121(pretrained=True)
            input_size = 1024
        elif structure == 'vgg16':
            model = models.vgg16(pretrained=True)
            input_size = 25088
        elif structure == 'vgg13':
            model = models.vgg13(pretrained=True)
            input_size = 4096
        
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        
        optimizer = optim.Adam(model.parameters(), checkpoint['learning_rate'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        
        print("Loaded checkpoint")
        
    print("Configured model.")
    
    
    #print(device)
    model.to(device)
    #return model.to(device)

def train(epochs = 3):
    epochs = epochs
    steps = 0
    print_every = 5
    print("Start to trainning...")
    #process = reset_process(epochs*(len(dataloaders.trainloader)))
    
    for e in range(epochs):
        running_loss = 0
    
        for inputs, labels in dataloaders.trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            steps += 1
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            #report_process(process, steps)
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, criterion)
                
                print(f"Epochs:{e+1}/{epochs}, ",
                      f"Training:{steps}/{len(dataloaders.trainloader)}, ",
                      f"Training loss:{'%.3f' %(running_loss/print_every)}.. ",
                      f"Validation loss: {'%.3f' %(valid_loss/len(dataloaders.validloader))}, ",
                      f"Validation accuracy: {'%.3f' %(accuracy/len(dataloaders.validloader))}, ")
        
       
    print("Total steps:", steps)
                    
def validation(model, criterion):
    valid_loss = 0
    accuracy = 0
    for inputs, labels in dataloaders.testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        log_ps = model(inputs).to(device)
        loss = criterion(log_ps, labels)
        valid_loss += loss.item()
                    
        ps = torch.exp(log_ps)
        top_ps, top_class = ps.topk(1, dim = 1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += equality.type(torch.FloatTensor).mean().item()
    return valid_loss, accuracy

def load_checkpoint(load_path = 'checkpoint.pth'):
    if os.path.isfile(load_path):
        print("Loading checkpoint")
        model = create_model(checkpoint = torch.load(load_path))
        
        #model = create_model(structure=checkpoint)    
        #optimizer.load_state_dict(checkpoint['optim_state_dict'])
    else:
        print(f"Can not find the file at {load_path}.\n Please input the right file path.")
        return _, _
    #return model, optimizer

def save_checkpoint(save_path, structure, hidden_units, dropout, lr, epochs):
    model.class_to_idx = image_datasets.train_data.class_to_idx
    #print(model)
    checkpoint = {'epochs': epochs,
                  'structure': structure,
                  'hidden_units': hidden_units,
                  'dropout': dropout,
                  'learning_rate': lr,
                  'state_dict' : model.state_dict(),
                  'optim_state_dict' : optimizer.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'classifier' : model.classifier
                 }
    #print(checkpoint)
    print("Saving checkpoint")
    torch.save(checkpoint, save_path)

def process_image(image_path):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    
    image = Image.open(image_path)
    
    
    return transform(image)

def predict(image, top_k = 1, cat_library_path = 'cat_to_name.json'):
    model.to(device)
                      
    with open(cat_library_path, 'r') as f:
        cat_to_name = json.load(f)
                      
    with torch.no_grad():
        
        model.eval()
        img = image.to(device)
        img.unsqueeze_(0)
        output = model(img)
        
        ps = torch.exp(output)
        
        top_ps, top_class = ps.topk(top_k, dim=1)
        top_ps, top_class = top_ps.cpu().numpy(), top_class.cpu().numpy()
        
        index_to_class = {x: y for y, x in model.class_to_idx.items()}
        #print(index_to_class)
        #top_class_id = [index_to_class[e] for e in top_class.ravel()]
        top_class_name = [cat_to_name[str(index_to_class[e])] for e in top_class.ravel()]
        #print(top_class_name)
        return top_ps, top_class, top_class_name
                  
                  
def reset_process(total):
    pbar = tqdm(total=total, desc="Transfer progress", ncols=100, position=0, leave=True)
    return pbar
                  
def report_process(pbar, process):
        pbar.update(process)
    


# In[313]:


# Debug
#if __name__ == "__main__":
    #load_data("flowers")
    #create_model("vgg13")


# In[314]:


if __name__ == "__main__":
    get_ipython().system('jupyter nbconvert --to python notebook.ipynb *.ipynb')


# In[ ]:





# In[ ]:




