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