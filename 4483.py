import torch 
import torchvision
from torchvision import transforms
from ImageFolder import ImageFolder
import os

TRAIN = False 

# 图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
data = ImageFolder(root='/media/mldadmin/home/s123mdg34_04/WangShengyuan/IE4483_furryyy/datasets', transform=transform)
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

# 加载预训练模型
model = torchvision.models.resnet50(pretrained=True)

# 替换最后一层全连接层
num_classes = len(data.classes)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)

# 模型训练
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 10

if TRAIN:
    for epoch in range(num_epochs):
        for inputs, labels in loader:
            print("here")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    model.save_state_dict(f'resnet50-10epoch-loss{loss.item():.4f}.pth')
else:
    if os.path.exists('resnet50-10epoch-loss0.0000.pth'):
        model.load_state_dict('resnet50-10epoch-loss0.0000.pth')
    print("skip training")

# 模型测试
model.eval()
model.to(device)
correct = 0
total = 0

# construct dataframe for output
import pandas as pd

res = {
    "img_path": [],
    "label": [],
    "pred": [],
    "logits": []
}

with torch.no_grad():
    for inputs, labels, paths in loader:
        filenames = [os.path.relpath(path, os.getcwd()) for path in paths]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        res["img_path"].extend(filenames)
        res["label"].extend(labels.cpu().numpy().tolist())
        res["pred"].extend(predicted.cpu().numpy().tolist())
        res["logits"].extend(outputs.cpu().numpy().tolist())

df = pd.DataFrame(res)
df.to_csv("result.csv", index=False)

print('Accuracy: {:.2f}%'.format(correct / total * 100))