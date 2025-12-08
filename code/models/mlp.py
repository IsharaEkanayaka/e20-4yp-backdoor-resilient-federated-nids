import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=100):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

        # Initialize weights to zero 
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x