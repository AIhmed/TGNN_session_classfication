import pandas as pd
import random
import time
from transformers import AutoTokenizer
from torch.nn import Embedding
from torch_geometric_temporal import DynamicGraphTemporalSignal
EMBEDDING_SIZE = 768

#  df2 = pd.DataFrame(pd.read_excel('ajCommentClassification.xlsx'))
#  df = pd.DataFrame(pd.read_excel('tweetClassificationSummary.xlsx'))
df = pd.DataFrame(pd.read_excel('labeledDataset.xlsx'))
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabert')
embeddings = Embedding(tokenizer.vocab_size, EMBEDDING_SIZE)
df = df.loc[~df['commentText'].isna()]
df = df.loc[~df['Label'].isna()]
df.reset_index(inplace=True)
df = df[:200]

print(len(df))

def max_comment_len(df):
    sentence_embedding = list()
    lengths = dict()
    for index, sample in enumerate(df.iterrows()):
        tokenized_inputs = tokenizer(sample[1]['commentText'], return_tensors='pt', truncation=True)['input_ids'][0]
        sentence_embedding.append(embeddings(tokenized_inputs))
        lengths[index] = tokenized_inputs.shape[0]

    return (lengths, sentence_embedding)


#  inside this i will create the graph connectivity (edges) in COO format
#  it is a matrix of [2 * num_edges] where each column in the two rows contain the connected node
#  the owner of the session only have incident connection to it. hence, we only see it's id on the second row.
#  for each of the node in the session have to have a connection with one node.
#  I also have to create the edge_indeces which contain information about the edges
#  for each graph we have a matrix of edge_indeces where each row repsent an edge and each column in that row represent the two node connected with that edge.

#  i also have to initialise the node feature which are the embedding for the tokens we have.


#  session_len = random.randint(10, 20)
#  seq_len, node_features = max_comment_len(df[: session_len])
#  g = [[[] for node in range(2)] for time_step in range(max(seq_len.values()))]
#  edge_weight = [[] for time_step in range(max(seq_len.values()))]


sequence_lengths, embedding_space = max_comment_len(df)
node_features = [[]for session in range(len(df)//10)]
targets = list()
g = [[[[] for j in range(2)] for seq in range(max(sequence_lengths.values()))] for session in range(len(df) // 10)]
edge_weights = [[[] for time_step in range(max(sequence_lengths.values()))] for session in range(len(df)//10)]


def attr_edge_weight(session, timestep, inc, adj):
    if df.loc[inc, 'Label'] == 'P' and df.loc[adj, 'Label'] == 'P':
        edge_weights[session][timestep].append(-1)
    elif df.loc[inc, 'Label'] == 'N' and df.loc[adj, 'Label'] == 'N':
        edge_weights[session][timestep].append(1)
    else:
        edge_weights[session][timestep].append(-1)


def create_temporal_graph_example(session, start, session_len):
    node_features[session] = embedding_space[start: start+session_len]
    owner = random.randint(1, session_len)
    for i in range(session_len):
        con = random.randint(1, session_len)
        print(f"creating graph at time step 0 for session {session}")
        print(f"creating connection for node number {i} with {con} for session {session}")
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

    for t in range(1, max(list(sequence_lengths.values())[start: start+session_len])):
        print(f" creating graph at time step {t}")
        g[session][t][0] = g[session][t-1][0].copy()
        g[session][t][1] = g[session][t-1][1].copy()
        edge_weights[session][t] = edge_weights[session][t-1].copy()
        print(f"the edge_weight at time step {t} is \n\n")
        print(edge_weights[session][t])
        print(f"the number of edge in the time step {t} is {len(g[session][t][0])} and {len(g[session][t][1])}")
        if t in list(sequence_lengths.values())[start: start + session_len]:
            node = list(sequence_lengths.keys())[start: start+session_len][list(sequence_lengths.values())[start: start+session_len].index(t)]
            for e in range(g[session][t][0].count(node)):
                index = g[session][t][0].index(node)
                g[session][t][0].pop(index)
                g[session][t][1].pop(index)
                edge_weights[session][t].pop(index)
            for e in range(g[session][t][1].count(node)):
                index = g[session][t][1].index(node)
                g[session][t][0].pop(index)
                g[session][t][1].pop(index)
                edge_weights[session][t].pop(index)
    if edge_weights[session][0].count(1) > edge_weights[session][0].count(-1):
        targets.append(1)
    else:
        targets.append(-1)


start = 0
s = 0  # s for session
while start < len(df)-20:
    session_len = random.randint(10, 20)
    create_temporal_graph_example(s, start, session_len)
    s += 1
    start += session_len
num_el = len(node_features)
for i in range(num_el):
    if i != len(targets):
        if node_features[-1] == []:
            del node_features[-1]
        if g[-1] == [[[] for j in range(2)] for seq in range(max(sequence_lengths.values()))]:
            del g[-1]
        if edge_weights[-1] == [[] for time_step in range(max(sequence_lengths.values()))]:
            del edge_weights[-1]
    else:
        break
print(node_features)
print(targets)
print(len(node_features), len(targets))
print(len(g), len(edge_weights))
data = DynamicGraphTemporalSignal(edge_indices=g, edge_weights=edge_weights, features=node_features, targets=targets)
print(data)
