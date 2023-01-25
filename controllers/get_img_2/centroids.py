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
from sklearn.cluster import KMeans
import collections

high_data_path = "/home/betul/Documents/my_project/controllers/get_img_2/data_4.npy"
high_data = np.load(high_data_path)
high_data = high_data+1
print(high_data.shape)

kmeans = KMeans(n_clusters=3, n_init="auto").fit(high_data)

counter = collections.Counter(kmeans.labels_)
print(counter)
centroids_2 = kmeans.cluster_centers_


#clf = K_Means()
#clf.fit(high_data)
#print(np.linalg.norm(clf.centroids[2]-centroids_2[0]))
print(np.linalg.norm(-centroids_2[0]))
#print(np.linalg.norm(clf.centroids[2]))

centroids_path = "/home/betul/Documents/my_project/controllers/get_img_2/cent_data.npy"
np.save(centroids_path, centroids_2)


