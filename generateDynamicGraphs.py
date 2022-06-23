import pandas as pd
import random
from transformers import AutoTokenizer
from torch.nn import Embedding
import torch
#  import numpy as np
import json
#  from torch_geometric_temporal import DynamicGraphTemporalSignal
EMBEDDING_SIZE = 256

df = pd.DataFrame(pd.read_excel('labeledDataset.xlsx'))
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabert')
embeddings = Embedding(tokenizer.vocab_size, EMBEDDING_SIZE)
df = df.loc[~df['commentText'].isna()]
df = df.loc[~df['Label'].isna()]
df.reset_index(inplace=True)
df = df[:200]


def max_comment_len(df):
    sentence_embedding = list()
    targets = list()
    lengths = dict()
    for index, sample in enumerate(df.iterrows()):
        tokenized_inputs = tokenizer(sample[1]['commentText'], return_tensors='pt', truncation=True)['input_ids'][0]
        sentence_embedding.append(tokenized_inputs.tolist())
        if sample[1]['Label'] == 'P':
            targets.append(-1)
        else:
            targets.append(1)
        lengths[index] = len(tokenized_inputs)

    return (lengths, sentence_embedding, targets)


sequence_lengths, embedding_space, labels = max_comment_len(df)
node_features = [[[] for timesteps in range(max(sequence_lengths.values()))] for session in range(len(df)//10)]
targets = [[[] for timestep in range(max(sequence_lengths.values()))] for session in range(len(df)//10)]
edge_indices = [[[[] for j in range(2)] for seq in range(max(sequence_lengths.values()))] for session in range(len(df) // 10)]
edge_weights = [[[] for timesteps in range(max(sequence_lengths.values()))] for session in range(len(df)//10)]


def attr_edge_weight(session, timestep, inc, adj):
    if df.loc[inc, 'Label'] == 'P' and df.loc[adj, 'Label'] == 'P':
        edge_weights[session][timestep].append(-1)
    elif df.loc[inc, 'Label'] == 'N' and df.loc[adj, 'Label'] == 'N':
        edge_weights[session][timestep].append(1)
    else:
        edge_weights[session][timestep].append(-1)


def create_temporal_graph_example(session, start, session_len):
    owner = random.randint(1, session_len)
    for i in range(session_len):
        con = random.randint(1, session_len)
        if i != con and i != owner:
            if con in edge_indices[session][0][0]:
                ind = edge_indices[session][0][0].index(con)
                if edge_indices[session][0][1][ind] != i:
                    edge_indices[session][0][0].append(i)
                    edge_indices[session][0][1].append(con)
                    attr_edge_weight(session, 0, start+i, start+con)
                else:
                    continue
            else:
                edge_indices[session][0][0].append(i)
                edge_indices[session][0][1].append(con)
                attr_edge_weight(session, 0, start+i, start+con)
    node_set = set(edge_indices[session][0][0]).union(set(edge_indices[session][0][1]))
    for node in node_set:
        node_features[session][0].append(embedding_space[start + node][0])
        targets[session][0].append(labels[start + node])

    for t in range(1, max(list(sequence_lengths.values())[start: start+session_len])):
        print(f"temporal graph number {session} and time step number {t}")
        for i in range(session_len):
            if len(embedding_space[start + i]) > t:  #  this is to loop through each node in the graph
                #  add the token corresponding to the comment and the time step if the comment is too long
                if i in edge_indices[session][t-1][0]:  #  if a connection with this node exist create one for this timestep
                    index = edge_indices[session][t-1][0].index(i)  #  get index of the node i in the first graph
                    neighbor = edge_indices[session][t-1][1][index]
                    if len(embedding_space[start + neighbor]) > t:
                        print(f"sending_node: {i},comment: {start+i}, comment_length: {len(embedding_space[start+i])}")
                        print(f"receiving_node: {index},comment: {start+index}, comment_length: {len(embedding_space[start+index])}")
                        print(f"adding edge_index ({i}, {neighbor})")
                        edge_indices[session][t][0].append(i)
                        edge_indices[session][t][1].append(edge_indices[session][t-1][1][index])  # find connection of i
                        edge_weights[session][t].append(edge_weights[session][t-1][index])
                #  now the problem is with node that only receive an edge, they will be added in the node lists

        node_set = set(edge_indices[session][t][0]).union(set(edge_indices[session][t][1]))
        print(f"the set of node existing in the session are")
        print(node_set)
        if len(node_set) != 0:
            for node in node_set:
                print(f"the number of token in the node {node} and comment {start + node} is {len(embedding_space[start + node])}")
                if len(embedding_space[start+node]) > t:
                    print(f"adding node {node} in the node_features")
                    node_features[session][t].append(embedding_space[start + node][t])
                    targets[session][t].append(labels[start + node])
            print(f"the number of nodes mentioned in the edge_indices at timestep {t}\t\t is {len(node_set)}")
            print(f"the number of nodes (tokens) in node_features at timestep {t} is \t\t{len(node_features[session][t])}")


start = 0
s = 0  # s for session
while start < len(df)-20:
    session_len = random.randint(10, 20)
    create_temporal_graph_example(s, start, session_len)
    s += 1
    start += session_len
num_el = len(node_features)

for i in range(num_el):
    if node_features[-1] == [[] for time_step in range(max(sequence_lengths.values()))]:
        del node_features[-1]
    if edge_indices[-1] == [[[] for j in range(2)] for seq in range(max(sequence_lengths.values()))]:
        del edge_indices[-1]
    if edge_weights[-1] == [[] for time_step in range(max(sequence_lengths.values()))]:
        del edge_weights[-1]

sample_data = list()
for i in range(len(node_features)):
    for j in range(node_features[i].count([])):
        node_features[i].remove([])
        targets[i].remove([])
        edge_weights[i].remove([])
        edge_indices[i].remove([[], []])
    sample_data.append({"session": i, "node_features": node_features[i], "edge_indices": edge_indices[i], "edge_weights": edge_weights[i], "targets": targets[i]})

false_count = list()
for example in sample_data:
    for t in range(len(example)):
        lf = len(example['node_features'][t])
        le = len(set(example['edge_indices'][t][0]).union(set(example['edge_indices'][t][1])))
        print(lf == le)
        false_count.append(lf == le)
        print(f"the number of nodes is node_features {lf}")
        print(f"the number of nodes is edge_indices {le}")
        print("\n\n")
print('\n\n\n')
print(false_count.count(False))
print(false_count.count(True))

f = open("sample_data2.json", "w")
json.dump(sample_data, f)
f.close()
