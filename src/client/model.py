import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_dim=72, num_classes=10):
        super(Net, self).__init__()
        
        # 4-Layer DNN (Deep Neural Network)
        # We use powers of 2 for neuron counts (standard practice)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(0.2) # Prevents overfitting

    def forward(self, x):
        # Layer 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Layer 2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Layer 3
        x = F.relu(self.fc3(x))
        
        # Output Layer (No Softmax here! PyTorch CrossEntropyLoss handles it)
        x = self.fc4(x)
        return x