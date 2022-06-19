import pandas as pd
import random
from transformers import AutoTokenizer
from torch.nn import Embedding
import numpy as np
import json
from torch_geometric_temporal import DynamicGraphTemporalSignal
EMBEDDING_SIZE = 768

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
g = [[[[] for j in range(2)] for seq in range(max(sequence_lengths.values()))] for session in range(len(df) // 10)]
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
            if con in g[session][0][0]:
                ind = g[session][0][0].index(con)
                if g[session][0][1][ind] != i:
                    g[session][0][0].append(i)
                    g[session][0][1].append(con)
                    attr_edge_weight(session, 0, start+i, start+con)
                else:
                    continue
            else:
                g[session][0][0].append(i)
                g[session][0][1].append(con)
                attr_edge_weight(session, 0, start+i, start+con)
        node_features[session][0].append(embedding_space[start+i][0])
        targets[session][0].append(labels[start + i])

    for t in range(1, max(list(sequence_lengths.values())[start: start+session_len])):
        print(f"temporal graph number {session} and time step number {t}")
        for i in range(session_len):
            targets[session][t].append(labels[start + i])
            print(f"the comment number {i} in the current session")
            if len(embedding_space[start + i]) > t:
                node_features[session][t].append(embedding_space[start + i][t])
                if i < len(edge_weights[session][0]):
                    edge_weights[session][t].append(edge_weights[session][0][i])
                    g[session][t][0].append(g[session][0][0][i])
                    g[session][t][1].append(g[session][0][1][i])


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
        if g[-1] == [[[] for j in range(2)] for seq in range(max(sequence_lengths.values()))]:
            del g[-1]
        if edge_weights[-1] == [[] for time_step in range(max(sequence_lengths.values()))]:
            del edge_weights[-1]

sample_data = list()
for i in range(len(node_features)):
    for j in range(node_features[i].count([])):
        node_features[i].remove([])
        targets[i].remove([])
        edge_weights[i].remove([])
        g[i].remove([[], []])


for s in node_features:
    for t in s:
        print(t)
data = DynamicGraphTemporalSignal(edge_indices=g[0], edge_weights=edge_weights[0], features=node_features[0], targets=targets[0])
example = next(iter(data))
