import pandas as pd

df = pd.DataFrame(pd.read_excel('tweetClassificationSummary.xlsx'))
df1 = pd.DataFrame(pd.read_excel('labeledDataset.xlsx'))
df2 = pd.DataFrame(pd.read_excel('ajCommentClassification.xlsx'))

with open('corpus.txt', 'a') as corpus:
    for sample in df['text']:
        corpus.write(sample)
        corpus.write('\n')
    for sample in df1.loc[~df1['commentText'].isna(), 'commentText']:
        corpus.write(sample)
        corpus.write('\n')
    for sample in df2['body']:
        corpus.write(sample)
        corpus.write('\n')
