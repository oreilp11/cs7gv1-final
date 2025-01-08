import torch.nn as nn
class SEBlock(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels//reduction_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels//reduction_rate, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _, = x.size()
        se = self.global_pool(x)
        se = se.view(b, c)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return x * se.view(b, c, 1, 1)