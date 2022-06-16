from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabert')
f = open('corpus.txt', 'r')
content = f.read()
f.close()
content = content.split('\n')
for i, text in enumerate(content[:30]):
    print(f"the {i}th comment is {text}")
