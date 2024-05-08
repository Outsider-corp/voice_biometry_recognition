import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNLayer(nn.Module):
    """ Time Delay Neural Network Layer (TDNN) """

    def __init__(self, input_dim, output_dim, context_size, dilation):
        super(TDNNLayer, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=context_size, dilation=dilation,
                              padding=context_size // 2)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)


class MeanStdPoolingLayer(nn.Module):
    """ A layer that pools mean and standard deviation """

    def forward(self, x):
        mean = torch.mean(x, dim=2, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=2, keepdim=True) + 1e-8)
        pooled = torch.cat((mean, std), dim=1)
        return pooled.squeeze(2)


class SpeakerEmbeddingNet(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(SpeakerEmbeddingNet, self).__init__()
        self.tdnn_layers = nn.ModuleList([
            TDNNLayer(feat_dim, 512, 5, 1),
            TDNNLayer(512, 512, 3, 2),
            TDNNLayer(512, 512, 3, 3),
            TDNNLayer(512, 512, 1, 1),
            TDNNLayer(512, 1500, 1, 1)
        ])
        self.pooling_layer = MeanStdPoolingLayer()
        self.embedding_layer = nn.Linear(3000,
                                         embedding_dim)  # Assuming pooling outputs 3000 features

    def forward(self, x):
        # Expect input x to be (batch, feat_dim, seq_len)
        x = x.permute(0, 2, 1)  # Convert to (batch, seq_len, feat_dim)
        for layer in self.tdnn_layers:
            x = layer(x)
        x = self.pooling_layer(x)
        embeddings = self.embedding_layer(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalization
        return embeddings


# Example usage:
model = SpeakerEmbeddingNet(feat_dim=30, embedding_dim=256)
print(model)
