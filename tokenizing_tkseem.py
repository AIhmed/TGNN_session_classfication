import pandas as pd
import tkseem as tk

df = pd.DataFrame(pd.read_excel("tweetClassificationSummary.xlsx"))

tokenizer = tk.RandomTokenizer()
tokenizer.train(file_path='corpus.txt')

tokens = tokenizer.tokenize(df.loc[0, 'text'])

print(df.loc[0, 'text'])
print(tokens)
