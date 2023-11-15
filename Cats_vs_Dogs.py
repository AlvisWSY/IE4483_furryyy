from ImageFolder import ImageFolder # Read images from directory

from tqdm import tqdm # Progress bar

import matplotlib.pyplot as plt # Plotting for testing

from torchsummary import summary # Model summary

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from torch.optim import SGD
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import resnet50,  ResNet50_Weights
import pandas as pd

#Load_Checkpoint
# Load_Checkpoint = True

# Learning_Rate
lr=1e-3

# Data_Transform, Train data processing: RandomRotation, Resize, Normalize
train_transform = transforms.Compose([
    transforms.RandomRotation((0,180), transforms.InterpolationMode.BILINEAR, expand=True),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_data = ImageFolder("Cats_vs_Dogs_Datasets/train", train_transform)
test_data = ImageFolder("Cats_vs_Dogs_Datasets/val", test_transform)
train_dataset = DataLoader(train_data, batch_size=256, shuffle=True)
test_dataset = DataLoader(test_data, batch_size=256, shuffle=False)

class Head(nn.Module):
    def __init__(self, in_features, out_features, Backbone):
        super().__init__()
        self.Backbone = Backbone
        self.linear = nn.Sequential(nn.ReLU(inplace=True),#Reasons for using ReLU: #1 preventing network overfitting #2 non-linear
                                    nn.Linear(in_features=in_features, out_features=out_features,bias=False))
    def forward(self, x):
        x = self.Backbone(x)
        x = self.linear(x)
        x = torch.squeeze(x, 1)
        return x
    
# Model Settings
model = Head(in_features=1000, out_features=2, Backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2))

# Device Settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)


df = pd.DataFrame(columns=['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])
num_epochs = 10

for epoch in range(num_epochs):
    train_acc = 0
    train_loss = 0
    model.train()
    for inputs, labels, direc in tqdm(train_dataset):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.cpu().detach().numpy()
        train_acc += torch.sum(torch.eq(torch.max(labels, 1).indices, torch.max(outputs, 1).indices)).cpu().detach().numpy()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataset)
    train_acc = train_acc/len(train_data)*100
    print('Epoch: {} | Train | Loss: {:.4f}, Acc: {:.2f}'.format(epoch, train_loss, train_acc))

    test_acc = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels, direc in tqdm(test_dataset):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.cpu().detach().numpy()
            test_acc += torch.sum(torch.eq(torch.max(labels, 1).indices, torch.max(outputs, 1).indices)).cpu().detach().numpy()
        test_loss /= len(test_dataset)
        test_acc = test_acc/len(test_data)*100
        print('Epoch: {} | Test | Loss: {:.4f}, Acc: {:.2f}'.format(epoch, test_loss, test_acc))
    # add the results to the dataframe
    df = df.append({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc, 'test_loss': test_loss}, ignore_index=True)

# save the dataframe as a csv file
filename = f"Results/CD/results_lr{lr}.csv"
df.to_csv(filename, index=False)
checkpoint_name = f"Checkpoints/CD/checkpoint_lr{lr}.pth"
torch.save(model, checkpoint_name)