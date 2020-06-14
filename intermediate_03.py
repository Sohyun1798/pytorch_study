import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

train_dataset = torchvision.datasets.MNIST(root='./', train=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST('./', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):

        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.
