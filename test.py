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
lr=1e-2

# Data_Transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_data = ImageFolder("data/Cats_vs_Dogs/test", test_transform)
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

model.load_state_dict(torch.load('Checkpoints/CD/checkpoint_lr0.001.pth'))

# Device Settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)


res = []

model.eval()
with torch.no_grad():
    for inputs, _, direcs in tqdm(test_dataset):
        inputs, _ = inputs.to(device), _.to(device)
        outputs = model(inputs)
        preds = torch.max(outputs, 1).indices
        for direc, pred in zip(direcs, preds.cpu().detach().numpy().tolist()):
            res.append({
                "img_dir": direc,
                "pred": pred
            })
# add the results to the dataframe
pd.DataFrame(res).to_csv('res.csv', index=False)



