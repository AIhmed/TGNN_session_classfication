import json
import torch
import torch.nn as nn


class SpatialEncoding:
    def __init__(self, example):
        self.embedding = nn.Embedding(64000, 128)
        self.node_features = example['node_features']
        self.edge_indices = example['edge_indices']
        self.edge_weights = example['edge_weights']
        self.targets = example['targets']

    def propagate(self, t):
        n = self.embedding(torch.tensor(self.node_features[t]))
        n_hat = n.clone()
        updated = list()
        for i in range(len(self.edge_indices[t][0])):
            if i not in updated:
                print(f"n_hat.shape: {n_hat.shape}")
                print(f"n.shape: {n.shape}")
                n_hat[self.edge_indices[t][1][i]] = n[self.edge_indices[t][1][i]] + n[self.edge_indices[t][0][i]] * self.edge_weights[t][i]
                updated.append(self.edge_indices[t][1][i])
            else:
                print(f"n_hat.shape: {n_hat.shape}")
                print(f"n.shape: {n.shape}")
                n_hat[self.edge_indices[t][1][i]] = n_hat[self.edge_indices[t][1][i]] + n[self.edge_indices[t][0][i]] * self.edge_weights[t][i]

        del n
        del updated
        return n_hat

    def __call__(self):
        t = torch.cat((self.propagate(0), self.propagate(1)))
        print(t.shape)
        for timestep in range(2, len(self.edge_indices)):
            print(f"the edge_indices we have for this timestep are {self.edge_indices[timestep]}")
            t = torch.cat((t, self.propagate(timestep)))
            print(f"shape of tensor at timestep {timestep} {t.shape}")
            print("\n\n")
        return t


f = open("sample_data2.json", "r")
data = json.load(f)
f.close()
spatial_encoding = SpatialEncoding(data[0])
node_embedding = spatial_encoding()
print(node_embedding.shape)
