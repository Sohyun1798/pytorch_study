# 01. autograd example

import torch
import torchvison
import torch.nn as nn
import numpy as np
import torchvison.transforms as transforms

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w * x + b

y.backward() # compute gradients

print(x.grad)
print(w.grad)
print(b.grad)

x = torch.randn(10, 3)
y = torch.randn(10, 2)

linear = nn.Linear(3,2)

print('w: ', linear.weight)
print('b: ', linear.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)

loss = criterion(pred, y)
print('loss: ', loss.item())

loss.backward()

print('dL/dw:', linear.weight.grad)
print('dL/dw:', linear.bias.grad)

optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimisation: ', loss.item())

x = np.array([[1,2], [3,4]])
y = torch.from_numpy(x)
z = y.numpy

train_dataset = torchvision.dataset.CIFAR10(root='../../data/', train=True, transform=transforms.ToTensor(), download=True)

image, label = train_dataset[0]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = 64, shuffle = True)

data_iter = iter(train_dataset)

images, labels = data_iter.next()

for images, labels in train_loader:
    pass

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # initialise file paths or a list of file names
        pass
    def __getitem__(self,index):
        # read one data from file(using numpy.fromfile, PIL.Image.open)
        # preprocess the data(torchvision.Transform)
        # return a datapair
        pass
    def __len__(self):
        return 0 # change 0 to total size of my dataset

custiom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,batch_size=64,shuffle=True)

resnet = torchvison.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.linear(resnet.fc.in_features, 100)

images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())

#save and load the entire model
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

#save and load only the model parameters
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
