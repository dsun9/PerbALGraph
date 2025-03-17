import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GAT, GCN, GIN, GraphSAGE


class GNN_Backbone(nn.Module):
    def __init__(self, layer_type, num_layers, in_channels, hidden_channels, out_channels, num_class, dropout=0.2):
        super().__init__()
        backbone_cls = {
            'gcn': GCN,
            'gat': GAT,
            'sage': GraphSAGE,
            'gin': GIN
        }[layer_type.lower()]
        self.num_class = num_class
        self.backbone = backbone_cls(in_channels, hidden_channels, num_layers, out_channels, dropout)
        self.fc = nn.Linear(out_channels, num_class)

    def forward(self, x, edge_index):
        z = self.backbone(x, edge_index)
        y = F.relu(z)
        y = self.fc(y)
        return z, F.softmax(y, dim=1)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
