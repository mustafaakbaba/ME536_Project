#import libraries
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
import matplotlib as plt
import numpy as np
from improved_k import K_Means
import cv2



high_data_path = "/home/betul/Documents/my_project/controllers/get_img_2/data_2.npy"
high_data = np.load(high_data_path)
centroids_path = "/home/betul/Documents/my_project/controllers/get_img_2/cent_data.npy"
centroids = np.load(centroids_path)
print(centroids.shape)

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

high_data = high_data+1

clf = K_Means()
clf.fit(high_data)
def predict_2(data):
    global centroid
    distances = [np.linalg.norm(data - centroids[centroid]) for centroid in range(3)]
    classification = distances.index(min(distances))
    print(distances)
    print(classification)
    return classification
    
def predict(test_image_name):
    global clf
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
        #print(out.shape)
        #print("Output class :  ", topclass, topk)
        class_num = topclass[0][0]
        #print(int(class_num))
        #print(topk)
        out = out.cpu().detach().numpy()
        down_im = np.copy(out)
        down_im = np.resize(down_im,(16,16))
        print(down_im.shape)
        #cv2.imshow(down_im)
        
        out = out/np.linalg.norm(out)
        out = out+1
        #predict_2(out)
        clf.predict(out)
        
        

        
        
 

"""
high_data = high_data+1
#print(S)
       
clf = K_Means()
clf.fit(high_data)
print(len(clf.centroids))
"""

img = cv2.imread('/home/betul/Documents/my_project/controllers/get_img/cnn/valid/duck/4.png')
img2 = cv2.imread('/home/betul/Documents/my_project/controllers/get_img/cnn/train/e-puck/45.png')
img3 = cv2.imread('/home/betul/Documents/my_project/controllers/get_img/cnn/test/paper_ship/2.png')
print("  1")
predict(img)



print("  2")
predict(img2)


print("  3")
predict(img3)

