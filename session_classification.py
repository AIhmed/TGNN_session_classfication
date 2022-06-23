import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import json
VOCAB_SIZE = 64000
EMBEDDING_SIZE = 256


class SpatialEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.gcn1 = gnn.GCNConv(in_channels=EMBEDDING_SIZE, out_channels=EMBEDDING_SIZE)
        self.gcn2 = gnn.GCNConv(in_channels=EMBEDDING_SIZE, out_channels=EMBEDDING_SIZE)
        self.gcn3 = gnn.GCNConv(in_channels=EMBEDDING_SIZE, out_channels=EMBEDDING_SIZE)

    def forward(self, node_features, edge_indices, edge_weights):
        edge_index0 = torch.tensor(edge_indices[0])
        edge_weight0 = torch.tensor(edge_weights[0], dtype=torch.float32)
        edge_index1 = torch.tensor(edge_indices[1])
        edge_weight1 = torch.tensor(edge_weights[1], dtype=torch.float32)
        s0 = torch.sparse_coo_tensor(edge_index0, edge_weight0)
        s1 = torch.sparse_coo_tensor(edge_index1, edge_weight1)

        print(len(node_features[0]))
        print(set(edge_indices[0][0]).union(set(edge_indices[0][1])))
        first = self.embedding(torch.tensor(node_features[0]))
        print(s0.to_dense().shape, first.shape)
        print(s0)
        print(first)

        first = self.gcn1(x=first, edge_index=edge_index0, edge_weight=edge_weight0)
        first = self.gcn2(x=first, edge_index=edge_index0, edge_weight=edge_weight0)
        first = self.gcn3(x=first, edge_index=edge_index0, edge_weight=edge_weight0)

        print(len(node_features[1]))
        print(set(edge_indices[1][0]).union(set(edge_indices[1][1])))
        second = self.embedding(torch.tensor(node_features[1]))
        print(s1.to_dense().shape, second.shape)

        second = self.gcn1(x=second, edge_index=edge_index1, edge_weight=edge_weight1)
        second = self.gcn2(x=second, edge_index=edge_index1, edge_weight=edge_weight1)
        second = self.gcn3(x=second, edge_index=edge_index1, edge_weight=edge_weight1)

        t = torch.cat((first, second))
        for i in range(2, len(node_features)):
            print("\n\n\n")
            i_edge_index = torch.tensor(edge_indices[i])
            i_edge_weight = torch.tensor(edge_weights[i], dtype=torch.float32)
            nth = self.embedding(torch.tensor(node_features[i]))
            nth = self.gcn1(x=nth, edge_index=i_edge_index, edge_weight=i_edge_weight)
            nth = self.gcn2(x=nth, edge_index=i_edge_index, edge_weight=i_edge_weight)
            nth = self.gcn3(x=nth, edge_index=i_edge_index, edge_weight=i_edge_weight)
            t = torch.cat((t, nth))
            print(t.shape)
        return t


f = open("sample_data2.json", "r")
data = json.load(f)
f.close()
spatial = SpatialEncoding()
num_tokens = sum([len(timestep) for timestep in data[0]['node_features']])
print(num_tokens)
out = spatial(data[0]['node_features'], data[0]['edge_indices'], data[0]['edge_weights'])
print(out.shape)
