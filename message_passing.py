import json
import torch
import torch.nn as nn
embedding = nn.Embedding(64000, 128)
f = open("sample_data3.json", "r")
data = json.load(f)
f.close()


class SpatialEncoding(nn.Module):
    def __init__(self):
        super(SpatialEncoding, self).__init__()
        self.projector = nn.Linear(128, 128)

    def propagate(self, t, x, shifted_edge_indices, edge_weights):
        # n = embedding(torch.tensor(node_features[t]))
        n_hat = x.clone()
        updated = list()
        for i in range(len(shifted_edge_indices[t][0])):
            if i not in updated:
                n_hat[shifted_edge_indices[t][1][i]] = x[shifted_edge_indices[t][1][i]] + x[shifted_edge_indices[t][0][i]] * edge_weights[t][i]
                updated.append(shifted_edge_indices[t][1][i])
            else:
                n_hat[shifted_edge_indices[t][1][i]] = n_hat[shifted_edge_indices[t][1][i]] + x[shifted_edge_indices[t][0][i]] * edge_weights[t][i]

        del updated
        return self.projector(n_hat)

    def forward(self, node_features, shifted_edge_indices, edge_weights):
        node_embedding = []
        for timestep in range(len(shifted_edge_indices)):
            x = embedding(torch.tensor(node_features[timestep]))
            x = self.propagate(timestep, x, shifted_edge_indices, edge_weights)
            x = self.propagate(timestep, x, shifted_edge_indices, edge_weights)
            node_embedding.append(self.propagate(timestep, x, shifted_edge_indices, edge_weights))
        return node_embedding


class TemporalEncoding(nn.Module):
    def __init__(self, example, input_size, hidden_size, num_heads):
        super(TemporalEncoding, self).__init__()
        self.example = example
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.nodes = set(example['edge_indices'][0][0]).union(set(example['edge_indices'][0][1]))
        self.spatial_encoding = SpatialEncoding()
        self.query = nn.Linear(input_size, self.num_heads * self.head_size)
        self.key = nn.Linear(input_size, self.num_heads * self.head_size)
        self.value = nn.Linear(input_size, self.num_heads * self.head_size)
        self.projector = nn.Linear(self.num_heads * self.head_size, 2)

    def concat_features(self, edge_indices, shifted_edge_indices, node_embedding):
        temporal_input = [[[] for timestep in range(len(node_embedding))] for i in range(len(self.nodes))]
        for node in self.nodes:
            for timestep in range(len(node_embedding)):
                if node in edge_indices[timestep][0]:
                    index = edge_indices[timestep][0].index(node)
                    n = shifted_edge_indices[timestep][0][index]
                    temporal_input[node][timestep] = node_embedding[timestep][n]
                elif node in edge_indices[timestep][1]:
                    index = edge_indices[timestep][1].index(node)
                    n = shifted_edge_indices[timestep][1][index]
                    temporal_input[node][timestep] = node_embedding[timestep][n]
                else:
                    break

        for node in self.nodes:
            t = 0
            while t < len(temporal_input[node]):
                if temporal_input[node][t] == []:
                    del temporal_input[node][t]
                else:
                    t += 1
        return temporal_input

    def split_to_attention_head(self, tensors):
        new_shape = tensors.shape[: 2] + (self.num_heads, self.head_size)  #  spliting features for each head
        tensors = tensors.view(new_shape)
        return tensors.transpose(1, 2)

    def compute_attention(self, node_features, edge_indices, shifted_edge_indices, edge_weights):
        node_embedding = self.spatial_encoding(node_features, shifted_edge_indices, edge_weights)
        temporal_inputs = self.concat_features(edge_indices, shifted_edge_indices, node_embedding)

        node_embedding = list()
        for node in self.nodes:
            t = torch.cat((temporal_inputs[node][0], temporal_inputs[node][1]))
            for timestep in range(2, len(temporal_inputs[node])):
                t = torch.cat((t, temporal_inputs[node][timestep]))
            t = t.reshape(1, -1, 128)

            query = self.split_to_attention_head(self.query(t))
            key = self.split_to_attention_head(self.key(t))
            value = self.split_to_attention_head(self.value(t))

            scores = query.matmul(key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_size))
            attention_scores = scores.softmax(dim=-1).matmul(value)
            attention_scores = attention_scores.reshape(1, t.shape[1], -1)
            node_embedding.append(attention_scores)

        return node_embedding

    def forward(self):
        output = self.compute_attention(self.example['node_features'],
                self.example['edge_indices'],
                self.example['shifted_edge_indices'],
                self.example['edge_weights'])
        agg = torch.cat((output[0][0], output[1][0]))
        for node in range(2, len(output)):
            agg = torch.cat((agg, output[node][0]))
        agg = agg.transpose(1, 0).mean(dim=-1).reshape(-1)
        t = self.projector(agg)
        return t

model = TemporalEncoding(data[0], 128, 128, 12)
output = model()
print(output.softmax(dim=-1).argmax(dim=-1))
