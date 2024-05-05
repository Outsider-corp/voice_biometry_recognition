import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNNLayer(nn.Module):
    """ Time Delay Neural Network Layer (TDNN) """
    def __init__(self, input_dim, output_dim, context_size, dilation):
        super(TDNNLayer, self).__init__()
        self.context_size = context_size
        self.dilation = dilation
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=self.context_size, dilation=self.dilation)

    def forward(self, x):
        # x shape: (batch, time, features) -> need to permute to (batch, features, time)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # Re-permute back to (batch, time, features)
        return x.permute(0, 2, 1)

class StatisticalPooling(nn.Module):
    """ Statistical Pooling Layer """
    def forward(self, x):
        # Compute mean and standard deviation
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        # Concatenate mean and std along feature dimension
        return torch.cat((mean, std), dim=1)

class XVector(nn.Module):
    def __init__(self, num_speakers):
        super(XVector, self).__init__()
        self.tdnn1 = TDNNLayer(input_dim=30, output_dim=512, context_size=5, dilation=1)
        self.tdnn2 = TDNNLayer(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.tdnn3 = TDNNLayer(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.tdnn4 = TDNNLayer(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.tdnn5 = TDNNLayer(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.pooling = StatisticalPooling()
        self.fc1 = nn.Linear(3000, 512)
        self.fc2 = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, num_speakers)

    def forward(self, x):
        x = self.tdnn1(x)
        x = F.relu(x)
        x = self.tdnn2(x)
        x = F.relu(x)
        x = self.tdnn3(x)
        x = F.relu(x)
        x = self.tdnn4(x)
        x = F.relu(x)
        x = self.tdnn5(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x

# Example usage:
num_speakers = 1000  # Adjust the number of speakers as required for your dataset
model = XVector(num_speakers=num_speakers)
print(model)
