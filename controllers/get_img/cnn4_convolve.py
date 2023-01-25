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


high_data = np.zeros((160,256))

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

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



# Load the Data
# Set train and valid directory paths
train_directory = '/home/betul/Documents/my_project/controllers/get_img/cnn/train'
valid_directory = '/home/betul/Documents/my_project/controllers/get_img/cnn/valid'
test_directory = '/home/betul/Documents/my_project/controllers/get_img/cnn/test'
# Batch size
bs = 32
# Number of classes
num_classes = 10
# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}
# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])
# Create iterators for the Data loaded using DataLoader module
train_data = DataLoader(data['train'], batch_size=bs, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=bs, shuffle=True)
test_data = DataLoader(data['test'], batch_size=bs, shuffle=True)
# Print the train, validation and test set data sizes
print(train_data_size, valid_data_size, test_data_size)


# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#transfer learning part
resnet50 = torchvision.models.resnet50(pretrained=True)
model = resnet50

# Freeze model parameters
for param in resnet50.parameters():
    param.requires_grad = False

# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 256, 3), 
    nn.LogSoftmax(dim=1) # For using NLLLoss()
)

# Convert model to be used on GPU
resnet50 = resnet50.to('cuda:0')

# Define Optimizer and Loss Function
loss_criterion = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())

#number of epochs
epochs = 20
history = []
early_stopper = EarlyStopper(patience=3, min_delta=.0010)


for epoch in range(epochs):
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch+1, epochs))
    # Set to training mode
    model.train()
    # Loss and Accuracy within the epoch
    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    temp = 0
    for i, (inputs, labels) in enumerate(train_data):
        #print(i)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Clean existing gradients
        optimizer.zero_grad()
        # Forward pass - compute outputs on input data using the model
        outputs = model(inputs)
        #print(outputs.shape)
        
        high_data[i*32:i*32+32,:] = outputs.cpu().detach().numpy()
        # Compute loss
        loss = loss_criterion(outputs, labels)
        # Backpropagate the gradients
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Compute the total loss for the batch and add it to train_loss
        train_loss += loss.item() * inputs.size(0)
        # Compute the accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        # Convert correct_counts to float and then compute the mean
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += acc.item() * inputs.size(0)
        #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
        # Validation - No gradient tracking needed
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()
            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)
                # Compute loss
                loss = loss_criterion(outputs, labels)
                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)
                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)
                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            # Find average training loss and training accuracy
            avg_train_loss = train_loss/train_data_size 
            avg_train_acc = train_acc/float(train_data_size)
            # Find average training loss and training accuracy
            avg_valid_loss = valid_loss/valid_data_size 
            avg_valid_acc = valid_acc/float(valid_data_size)
            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
            epoch_end = time.time()
            print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, nttValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        """if early_stopper.early_stop(valid_loss): 
            print(" ")
            print("early stop", epoch)            
            break"""





## save the model
model_path = '/home/betul/Documents/my_project/controllers/get_img/cnn/model/model_conv.pth'
torch.save(model, model_path)

high_data_path = "/home/betul/Documents/my_project/controllers/get_img_2/data_2.npy"
np.save(high_data_path, high_data)



