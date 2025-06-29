# Define Policy Network with GNN
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class PolicyGNN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(PolicyGNN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_features)

    def forward(self, x, edge_index):
        # Use multiple GCN layers for better graph understanding
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return torch.softmax(x, dim=-1)  # Output action probabilities
