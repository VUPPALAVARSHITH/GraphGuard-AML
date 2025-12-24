"""
model_architecture.py

GraphSAGE + GAT model for transaction-level AML risk prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class GraphSAGE_GAT(nn.Module):
    def __init__(self, num_nodes, edge_feat_dim):
        super().__init__()

        # Node embeddings
        self.node_emb = nn.Embedding(num_nodes, 32)

        # GraphSAGE layers
        self.sage1 = SAGEConv(32, 32)
        self.sage2 = SAGEConv(32, 32)

        # GAT layer
        self.gat = GATConv(32, 32, heads=2, concat=False)

        # Edge-level MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(32 * 2 + edge_feat_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, edge_index, edge_attr):
        x = self.node_emb.weight

        x = F.relu(self.sage1(x, edge_index))
        x = F.relu(self.sage2(x, edge_index))
        x = F.relu(self.gat(x, edge_index))

        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=1)

        return self.edge_mlp(edge_input).squeeze()
