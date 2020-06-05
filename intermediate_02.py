import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 80
batch_size = 100
learning_rate = 0.001

transform = transforms.Compose([transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)