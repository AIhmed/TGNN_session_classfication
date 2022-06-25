import pandas as pd
import random
from transformers import AutoTokenizer
from torch.nn import Embedding
import json
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
shifted_edge_indices = [[[[] for j in range(2)] for seq in range(max(sequence_lengths.values()))] for session in range(len(df) // 10)]
edge_weights = [[[] for timesteps in range(max(sequence_lengths.values()))] for session in range(len(df)//10)]


def attr_edge_weight(session, timestep, inc, adj):
    if df.loc[inc, 'Label'] == 'P' and df.loc[adj, 'Label'] == 'P':
        edge_weights[session][timestep].append(-1)
    elif df.loc[inc, 'Label'] == 'N' and df.loc[adj, 'Label'] == 'N':
        edge_weights[session][timestep].append(1)
    else:
        edge_weights[session][timestep].append(-1)


def create_temporal_graph_example(session, start, session_len):
    owner = random.randint(1, session_len-1)
    for i in range(session_len):
        node_features[session][0].append(embedding_space[start + i][0])
        targets[session][0].append(labels[start + i])
        con = random.randint(1, session_len-1)
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

    all_nodes = set(range(len(node_features[session][0])))
    edge_nodes = set(edge_indices[session][0][0]).union(set(edge_indices[session][0][1]))
    isolated_nodes = all_nodes - edge_nodes
    for nodes in isolated_nodes:
        edge_indices[session][0][0].append(nodes)
        edge_indices[session][0][1].append(nodes)
        edge_weights[session][0].append(0)
    shifted_edge_indices[session][0] = edge_indices[session][0].copy()
    print(f"the edge_indices for temporal graph {session} and timestep 0 is")
    print(edge_indices[session][0])
    print(shifted_edge_indices[session][0])
    for t in range(1, max(list(sequence_lengths.values())[start: start+session_len])):
        print(f"temporal graph number {session} and time step number {t}")
        for i in range(session_len):
            if len(embedding_space[start + i]) > t:  #  this is to loop through each node in the graph
                node_features[session][t].append(embedding_space[start + i][t])
                targets[session][t].append(labels[start + i])
                #  add the token corresponding to the comment and the time step if the comment is too long
                if i in edge_indices[session][t-1][0]:  #  if a connection with this node exist create one for this timestep
                    index = edge_indices[session][t-1][0].index(i)  #  get index of the node i in the first graph
                    neighbor = edge_indices[session][t-1][1][index]
                    if len(embedding_space[start + neighbor]) > t:
                        print(f"sending_node: {i},comment: {start+i}, comment_length: {len(embedding_space[start+i])}")
                        print(f"receiving_node: {neighbor},comment: {start+neighbor}, comment_length: {len(embedding_space[start+neighbor])}")
                        print(f"adding edge_index ({i}, {neighbor})")
                        edge_indices[session][t][0].append(i)
                        edge_indices[session][t][1].append(neighbor)
                        edge_weights[session][t].append(edge_weights[session][t-1][index])
                    else:
                        edge_indices[session][t][0].append(i)
                        edge_indices[session][t][1].append(i)
                        edge_weights[session][t].append(0)

            #  adding isolated nodes is important for counting number of nodes in the graph at a certain timestep
            else:
                if i in edge_indices[session][t-1][0]:
                    index = edge_indices[session][t-1][0].index(i)
                    neighbor = edge_indices[session][t-1][1][index]
                    if len(embedding_space[start + neighbor]) > t:
                        if neighbor not in edge_indices[session][t-1][0]:
                            edge_indices[session][t][0].append(neighbor)
                            edge_indices[session][t][1].append(neighbor)
                            edge_weights[session][t].append(0)
                if i in edge_indices[session][t-1][1]:
                    index = edge_indices[session][t-1][1].index(i)
                    sender = edge_indices[session][t-1][0][index]
                    if len(embedding_space[start + sender]) > t:
                        if sender not in edge_indices[session][t-1][1]:
                            edge_indices[session][t][0].append(sender)
                            edge_indices[session][t][1].append(sender)
                            edge_weights[session][t].append(0)

        edge_nodes = set(edge_indices[session][t][0]).union(set(edge_indices[session][t][1]))
        f_nodes = list(range(len(node_features[session][t])))
        shifted_nodes = dict()
        for i, n in enumerate(edge_nodes):
            shifted_nodes[n] = f_nodes[i]
        for i, e in enumerate(edge_indices[session][t][0]):
            shifted_edge_indices[session][t][0].append(shifted_nodes[e])
            shifted_edge_indices[session][t][1].append(shifted_nodes[edge_indices[session][t][1][i]])
        print("this shifted nodes are")
        print(shifted_nodes)
        print(f"the edge_indices for temporal graph {session} and timestep {t} is")
        print(edge_indices[session][t])
        print(shifted_edge_indices[session][t])
        print("\n\n")


start = 0
s = 0  # s for session
while start < len(df)-20:
    session_len = random.randint(10, 20)
    create_temporal_graph_example(s, start, session_len)
    s += 1
    start += session_len

i = 0
while i < len(node_features):
    b0 = node_features[i] == [[] for time_step in range(max(sequence_lengths.values()))]
    b1 = targets[i] == [[] for timestep in range(max(sequence_lengths.values()))]
    if b0 and b1:
        del node_features[i]
        del targets[i]
    else:
        i += 1
i = 0
while i < len(edge_indices):
    b0 = edge_indices[i] == [[[] for j in range(2)] for seq in range(max(sequence_lengths.values()))]
    b1 = shifted_edge_indices[i] == [[[] for j in range(2)] for seq in range(max(sequence_lengths.values()))]
    b2 = edge_weights[i] == [[] for time_step in range(max(sequence_lengths.values()))]
    if b0 and b1 and b2:
        del edge_indices[i]
        del shifted_edge_indices[i]
        del edge_weights[i]
    else:
        i += 1

sample_data = list()
for i in range(len(node_features)):
    for j in range(node_features[i].count([])):
        node_features[i].remove([])
        targets[i].remove([])
    for j in range(edge_weights[i].count([])):
        edge_weights[i].remove([])
        edge_indices[i].remove([[], []])
        shifted_edge_indices[i].remove([[], []])
    sample_data.append({"session": i, "node_features": node_features[i], "edge_indices": edge_indices[i], "shifted_edge_indices": shifted_edge_indices[i], "edge_weights": edge_weights[i], "targets": targets[i]})
example = sample_data[0]
for i, e in enumerate(example['edge_indices']):
    print(e)
    print(example['shifted_edge_indices'][i])
    print(example['node_features'][i])
    print("\n\n\n")
f = open("sample_data3.json", "w")
json.dump(sample_data, f)
f.close()
