import pandas as pd
import random
import time

#df = pd.DataFrame(pd.read_excel('ajCommentClassification.xlsx'))
#df1 = pd.DataFrame(pd.read_excel('labeledDataset.xlsx'))
df = pd.DataFrame(pd.read_excel('tweetClassificationSummary.xlsx'))


def max_comment_len(df):
    max = 0
    for sample in df.iterrows():
        if len(sample[1]['text']) > max:
            max = len(sample[1]['text'])
    return max


start = 0
end = 0
s = 0  # s for session
max_session = 20
min_session = 10
max_seq_len = max_comment_len(df)
dataset = [[[[] for j in range(20)] for seq in range(max_seq_len)] for i in range(len(df) // 10)]

while end < len(df):
    session_len = random.randint(min_session, max_session)
    start = end
    end += session_len
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
print(len(dataset[0]))
print(len(dataset[0][0]))
