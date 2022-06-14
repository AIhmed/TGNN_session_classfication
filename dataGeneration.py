import pandas as pd
import random
import time

#df = pd.DataFrame(pd.read_excel('ajCommentClassification.xlsx'))
#df1 = pd.DataFrame(pd.read_excel('labeledDataset.xlsx'))
df = pd.DataFrame(pd.read_excel('tweetClassificationSummary.xlsx'))


def max_comment_len(df):
    lengths = dict()
    for index, sample in enumerate(df.iterrows()):
        lengths[index] = len(sample[1]['text'].split(' '))
    return lengths


start = 0
end = 0
s = 0  # s for session
max_session = 20
min_session = 10
max_seq_len = max_comment_len(df)
#dataset = [[[[] for j in range(2)] for seq in range(max_seq_len['max'])] for session in range(len(df) // 10)]

#  I also have to create the edge_indeces which contain information about the edges
#  for each graph we have a matrix of edge_indeces where each row repsent an edge and each column in that row represent the two node connected with that edge.

#  i also have to initialise the node feature which are the embedding for the tokens we have.
'''
while end < len(df):
    session_len = random.randint(min_session, max_session)
    start = end
    end += session_len
    s += 1
    session_owner = random.randint(1, session_len)
    max_seq_len = max_comment_len(df[start:end])
    for t in range(max_seq_len):
        for i in range(session_len):
            if i == session_owner:
                for j in range(session_len):
                    dataset[s][t][i].append(0)
            else:
                con = random.randint(1, session_len)
                for j in range(session_len):
                    if j == con:
                        dataset[s][t][i].append(1)
                    else:
                        dataset[s][t][i].append(0)
        del dataset[s][t][i][-1]
        print(f"the starting position is {start} and then ending position is {end}\n and the max comment length is {max_seq_len}")
        print(dataset[s][t])

print("printing the first session\n\n\n\n\n\n\n\n")
print(dataset[100][0])
'''

session_len = random.randint(10, 20)
seq_len = max_comment_len(df[0: session_len])
g = [[[] for node in range(2)] for time_step in range(max(seq_len.values()))]
print(seq_len)


def create_temporal_graph_example(start, session_len, seq_len):
    owner = random.randint(1, session_len)
    for i in range(session_len):
        con = random.randint(1, session_len)
        print("creating graph at time step 0")
        print(f"creating connection for node number {i} with {con}")
        if i != con and i != owner:
            if con in g[0][0]:
                ind = g[0][0].index(con)
                if g[0][1][ind] != i:
                    g[0][0].append(i)
                    g[0][1].append(con)
                else:
                    continue
            else:
                g[0][0].append(i)
                g[0][1].append(con)

    for t in range(1, max(seq_len.values())):
        print(f" creating graph at time step {t}")
        #  inside this i will create the graph connectivity (edges) in COO format
        #  it is a matrix of [2 * num_edges] where each column in the two rows contain the connected node
        #  the owner of the session only have incident connection to it. hence, we only see it's id on the second row.
        #  for each of the node in the session have to have a connection with one node. 
        g[t][0] = g[t-1][0].copy()
        g[t][1] = g[t-1][1].copy()
        print(f"the number of edge in the time step {t} is {len(g[t][0])} and {len(g[t][1])}")
        if t in seq_len.values():
            node = list(seq_len.keys())[list(seq_len.values()).index(t)]
            for e in range(g[t][0].count(node)):
                index = g[t][0].index(node)
                g[t][0].pop(index)
                g[t][1].pop(index)
            for e in range(g[t][1].count(node)):
                index = g[t][1].index(node)
                g[t][0].pop(index)
                g[t][1].pop(index)


create_temporal_graph_example(0, session_len, seq_len)
for t in range(max(seq_len.values())):
    print(f"the edge indeces matrix for the time step {t} is: \n\n\t{g[t]}\n")
