import torch
from torch_geometric import utils
from torch_geometric.utils import scatter, unbatch_edge_index, dense_to_sparse

from torch_geometric.nn import GCNConv, DMoNPooling, DenseGraphConv
from math import ceil

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels = 4096, hidden_channels = 512, pool_size = 0.5, avg_num_nodes = 12, out_dim = 5):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.num_nodes = avg_num_nodes
        self.pool = DMoNPooling([hidden_channels, hidden_channels], self.num_nodes)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_dim)
        

    def forward(self, x, edge_index, batch):
        #pdb.set_trace()
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        #x = self.conv2(x, edge_index)
        x, mask = utils.to_dense_batch(x, batch)
        adj = utils.to_dense_adj(edge_index, batch)

        _, x, adj, sp1, o1, c1 = self.pool(x, adj, mask) #self.pool(x, edge_index, batch.batch)
        x = self.conv2(x, adj)
        #x = global_mean_pool(x.squeeze(), batch)
        #pdb.set_trace()
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x, c1 + o1 + sp1
    
    def train_step(self, batch, lang_model):
        x, edge_index, attention_mask = batch.x, batch.edge_index, batch.attention_mask
        #pdb.set_trace()
        inputs = lang_model(x, output_hidden_states=True, attention_mask=attention_mask)
        hidden_states = inputs.hidden_states
        last_hidden_state = hidden_states[-1]
        denom = torch.sum(attention_mask, -1, keepdim=True)
        feat = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / denom
        feat = feat.to(torch.float32)
        out, loss = self(feat, utils.to_undirected(edge_index), batch.batch)
        return out, loss