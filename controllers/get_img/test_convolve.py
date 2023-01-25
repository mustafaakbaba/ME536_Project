import matplotlib as plt
import torch
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn, optim
import time
import matplotlib.pyplot as plt
import numpy as np
import io


# Applying Transforms to the Data
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
}
"""
idx_to_class = {
    1: "ball",
    2: "duck",
    3: "paper_ship",
    4: "wheel",
}"""        
        

def predict(test_image_name):
    #load the model
    PATH = '/home/betul/Documents/my_project/controllers/get_img/cnn/model/model_conv.pth'
    model = torch.load(PATH)
    model.eval()
    
    transform = image_transforms['test']
    #test_image = Image.open(test_image_name)
    test_image = Image.fromarray(np.uint8(test_image_name)).convert('RGB')
    #test_image = test_image_name
    #plt.imshow(test_image)
    #plt.show()
    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        topclass = topclass.cpu()
        print(ps.shape)
        #print("Output class :  ", topclass, topk)
        class_num = topclass[0][0]
        #print(int(class_num))
        #print(topk)



"""
#load the model
PATH = '/home/betul/Documents/my_project/controllers/get_img/cnn/model/model.pth'
model = torch.load(PATH)
model.eval()


im_path = '/home/betul/Documents/my_project/controllers/get_img/cnn/test/ball/1.png'
predict(model, im_path)"""
