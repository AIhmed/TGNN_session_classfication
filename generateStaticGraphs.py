import pandas as pd
import random
from transformers import AutoTokenizer
from torch.nn import Embedding
from torch_geometric_temporal import StaticGraphTemporalSignal
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
    lengths = dict()
    for index, sample in enumerate(df.iterrows()):
        tokenized_inputs = tokenizer(sample[1]['commentText'], return_tensors='pt', truncation=True)['input_ids'][0]
        sentence_embedding.append(embeddings(tokenized_inputs))
        lengths[index] = tokenized_inputs.shape[0]

    return (lengths, sentence_embedding)


sequence_lengths, embedding_space = max_comment_len(df)
node_features = [[[] for timesteps in range(max(sequence_lengths.values()))] for session in range(len(df)//10)]
targets = list()
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

    for t in range(1, max(list(sequence_lengths.values())[start: start+session_len])):
        g[session][t][0] = g[session][t-1][0].copy()
        g[session][t][1] = g[session][t-1][1].copy()
        edge_weights[session][t] = edge_weights[session][t-1].copy()
        while i < session_len:
            if len(embedding_space[start + i]) > t:
                node_features[session][t].append(embedding_space[start + i][t])
            else:
                edge_weights[session][t][i] = 0
            i += 1
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
    if node_features[-1] == [[] for time_step in range(max(sequence_lengths.values()))]:
        del node_features[-1]
    if g[-1] == [[[] for j in range(2)] for seq in range(max(sequence_lengths.values()))]:
        del g[-1]
    if edge_weights[-1] == [[] for time_step in range(max(sequence_lengths.values()))]:
        del edge_weights[-1]
for i in range(len(edge_weights)):
    print(edge_weights[i])
