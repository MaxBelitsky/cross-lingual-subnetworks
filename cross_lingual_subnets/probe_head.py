import torch
from torch import nn

class ProbingHead(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_labels, pooling='cls'):
        super().__init__()
        self.pooling = pooling
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        if self.pooling == 'cls':
            x = features[:, 0, :]
        elif self.pooling == 'mean':
            x = torch.mean(features, dim=1)
        
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x

def train_probe():
