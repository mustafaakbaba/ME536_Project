#import libraries
import torch
from PIL import Image
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn, optim
import time
import matplotlib.pyplot as plt
import numpy as np
from improved_k import K_Means
import cv2 as cv
import torch.nn.functional as F


high_data_path = "/home/betul/Documents/my_project/controllers/get_img_2/data_3.npy"
high_data = np.load(high_data_path)
centroids_path = "/home/betul/Documents/my_project/controllers/get_img_2/cent_data.npy"
centroids = np.load(centroids_path)
#print(centroids.shape)

# load the model
model = models.resnet50(pretrained=True)
#print(model)
model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list
# get all the model children as list
model_children = list(model.children())

# counter to keep count of the conv layers
counter = 0 

# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)

# Applying Transforms to the Data
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

high_data = high_data + 1

clf = K_Means()
clf.fit(high_data)
def predict_2(data):
    global centroid
    distances = [np.linalg.norm(data - centroids[centroid]) for centroid in range(3)]
    classification = distances.index(min(distances))
    print(distances)
    print(classification)
    return classification
    
def predict(img):
    global clf
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    img = img.unsqueeze(0)
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results
    last_layer = outputs[-1][0,:,:,:]
    data = F.relu(last_layer)
    data = torch.flatten(data)
    data = data.cpu().detach().numpy()
    last_layer = last_layer.cpu().detach().numpy()
    for i, filter in enumerate(last_layer):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        #plt.subplot(8, 8, i + 1)
        #plt.imshow(filter, cmap='gray')
        #print(filter.shape)
        #plt.axis("off")
    #print(f"Saving layer {num_layer} feature maps...")
    #plt.savefig(f"../outputs/layer_{num_layer}.png")
    plt.show()
    plt.close()
    

    #out = out/np.linalg.norm(out)
    #out = out+1
    data = data + 1
    #predict_2(data)
    clf.predict(data)
        
        

        
        
 
#print(high_data[150,10000])
"""
high_data = high_data+1
#print(S)
       
clf = K_Means()
clf.fit(high_data)
print(len(clf.centroids))
"""

img = cv.imread('/home/betul/Documents/my_project/controllers/get_img/cnn/valid/duck/4.png')
img2 = cv.imread('/home/betul/Documents/my_project/controllers/get_img/cnn/train/e-puck/45.png')
img3 = cv.imread('/home/betul/Documents/my_project/controllers/get_img/cnn/test/paper_ship/2.png')
print("  1")
predict(img)



print("  2")
predict(img2)


print("  3")
predict(img3)

