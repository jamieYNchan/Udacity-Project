#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[22]:


# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

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


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[47]:


# TODO: Define your transforms for the training, validation, and testing sets


data_transforms = {'train_transforms' : transforms.Compose([transforms.RandomRotation(30),
                                                            transforms.RandomResizedCrop(224),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                                 [0.229, 0.224, 0.225])]), 
                   'test_transforms' : transforms.Compose([transforms.Resize(255),
                                                           transforms.CenterCrop(244),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                                [0.229, 0.224, 0.225])]),
                   'validation_transforms': transforms.Compose([transforms.Resize(255),
                                                                transforms.CenterCrop(244),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225])])
                  }



# TODO: Load the datasets with ImageFolder
image_datasets = {'train_data' : datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
                  'test_data' : datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms']), 
                  'valid_data' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation_transforms'])
                 }


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'trainloader' : torch.utils.data.DataLoader(image_datasets['train_data'], batch_size = 64, shuffle=True),
               'testloader' : torch.utils.data.DataLoader(image_datasets['test_data'], batch_size = 32),
               'validloader' : torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size = 32)}

print(f"trainloader size: {len(image_datasets['train_data'])}" , f" testloader size: {len(image_datasets['test_data'])}")


# In[ ]:





# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[71]:


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    print("Data: ", cat_to_name)
    print("Length:", len(cat_to_name))


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[6]:


# TODO: Build and train your network

def createDensenet121Model(checkpoint = None):
    

    model = models.densenet121(pretrained=True) #in : 1024
    #model = models.vgg16(pretrained=True) #in : 25088
    print(f"model: {model}")
    
    features_count = model.classifier.in_features
    print(f"features count: {features_count}")

    for param in model.parameters():
        param.requires_grad = False
    
    if checkpoint is None:
        model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024, 512)),
                                                      ('relu1', nn.ReLU()),
                                                      ('dropout1', nn.Dropout(p = 0.3)),
                                                      ('fc2', nn.Linear(512, 102)),
                                                      ('output', nn.LogSoftmax(dim = 1))
                                                     ]))
    else:
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
            
    return model

model = createDensenet121Model()
#model


# In[7]:


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

model.to(device)


# In[8]:


def validation():
    valid_loss = 0
    accuracy = 0
    for inputs, labels in dataloaders['testloader']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    log_ps = model(inputs)
                    loss = criterion(log_ps, labels)
                    valid_loss += loss.item()
                    
                    ps = torch.exp(log_ps)
                    top_ps, top_class = ps.topk(1, dim = 1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += equality.type(torch.FloatTensor).mean().item()
                    

    return valid_loss, accuracy


# In[9]:


epochs = 3
steps = 0
print_every = 5

for e in range(epochs):
    start = time.time()
    running_loss = 0
    
    for inputs, labels in dataloaders['trainloader']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        steps += 1
        
        log_ps = model.forward(inputs)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                valid_loss, accuracy = validation()
            
            print(f"Epochs:{e+1}/{epochs}",  
                  f"Training loss:{'%.3f' %(running_loss/print_every)}",
                  f"Validation loss: {'%.3f' %(valid_loss/len(dataloaders['validloader']))}",
                  f"Validation accuracy: {'%.3f' %(accuracy/len(dataloaders['validloader']))}")
                    
        


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[70]:


# TODO: Do validation on the test set

with torch.no_grad():
    accuracy = 0;
    model.eval()
    for inputs, labels in dataloaders['testloader']:
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        ps = torch.exp(output)
        
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += equality.type(torch.FloatTensor).mean().item()
        
print(f"Testing Accuracy: {'%.3f' %(accuracy/len(dataloaders['testloader']))}")
        


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[12]:


# TODO: Save the checkpoint 
#print(image_datasets['train_data'].class_to_idx)
model.class_to_idx = image_datasets['train_data'].class_to_idx
#print(model)
checkpoint = {'epoch': epochs,
              'state_dict' : model.state_dict(),
              'optim_state_dict' : optimizer.state_dict(),
              'class_to_idx' : model.class_to_idx,
              'classifier' : model.classifier
             }

torch.save(checkpoint, 'checkpoint.pth')
#print(model)
#print(model.state_dict().keys())
#print(model.class_to_idx)
#print(model.state_dict())


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[52]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
def loadCheckpoint(filePath):
    if os.path.isfile(filePath):
        checkpoint = torch.load(filePath)
        print(checkpoint)
        if not checkpoint is None:
        
            model = createDensenet121Model(checkpoint)
            
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
    
        return model, optimizer
    


# In[53]:


model, optimizer = loadCheckpoint('checkpoint.pth')

#print(model)


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[58]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    
    image = Image.open(image)
    
    
    return transform(image)


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[167]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if not title is None:
        ax.axis('off')
        ax.set_title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[151]:


#Test

imgPath = data_dir + '/test' + '/1' + '/image_06743.jpg'
imgTest = process_image(imgPath)
print(imgTest.shape)
imshow(imgTest)


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[152]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    model.to(device)
    
    with torch.no_grad():
        
        model.eval()
        img = process_image(image_path).to(device)
        img.unsqueeze_(0)
        output = model(img)
        
        ps = torch.exp(output)
        
        top_ps, top_class = ps.topk(topk, dim=1)
        top_ps, top_class = top_ps.cpu().numpy(), top_class.cpu().numpy()
        
        index_to_class = {x: y for y, x in model.class_to_idx.items()}
        #print(index_to_class)
        #top_class_id = [index_to_class[e] for e in top_class.ravel()]
        top_class_name = [cat_to_name[str(index_to_class[e])] for e in top_class.ravel()]
        #print(top_class_name)
        return top_ps, top_class, top_class_name, img

probs, classes,_,_ = predict(imgPath, model)


# In[153]:


probs, classes, classesName,_ = predict(imgPath, model)
print(probs)
print(classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[268]:


def imShowWithPredict(imgPath, model):
    
    probs, classes, classesName, image = predict(imgPath, model)
    #print(probs.ravel().tolist())
    #print(np.arange(0, 1, 0.1))
    image = image.squeeze(0)
    
    fig, ax = plt.subplots(nrows=2)
    ax[0] = imshow(image.cpu(), ax=ax[0], title=classesName[0])
    
    x_range = np.around(np.arange(0, 1.2, 0.2), decimals=1)
    ax[1].set_yticklabels(classesName)
    ax[1].set_xticklabels(x_range)
    ax[1].set_xlim(0, 1)
    ax[1].barh(classesName, probs.ravel().tolist())
    
#imgPath = data_dir + '/test' + '/19' + '/image_06197.jpg'


# In[269]:


# TODO: Display an image along with the top 5 classes

imgPath = data_dir + '/images.jpg'
imShowWithPredict(imgPath, model)


# # Reference
# 
# #### Torchvision
# 
# https://blog.csdn.net/u014380165/article/details/78634829
# 
# 
# #### Dictionary
# https://www.mkyong.com/python/python-how-to-loop-a-dictionary/
# https://stackoverflow.com/questions/8305518/switching-keys-and-values-in-a-dictionary-in-python
# 
# 
# #### Draw chart
# https://matplotlib.org/3.1.1/api/axes_api.html
# https://stackoverflow.com/questions/52171321/creating-alternative-y-axis-labels-for-a-horizontal-bar-plot-with-matplotlib
# https://stackoverflow.com/questions/7545015/can-someone-explain-this-0-2-0-1-0-30000000000000004
# 
# #### Common
# https://blog.csdn.net/wangdingqiaoit/article/details/41253947
# 

# In[270]:


if __name__ == "__main__":
    get_ipython().getoutput('jupyter nbconvert --to html notebook.ipynb *.ipynb')

