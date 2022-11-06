import pandas as pd

for a, b in zip(['train.tsv', 'dev.tsv'], ['train.txt', 'dev.txt']):
    with open(b, 'w', encoding='utf-8') as fp:
        df = pd.read_csv(a, sep='\t')
        for _, row in df.iterrows():
            label = row['label']
            text = row['text_a']
            print(str(label) + '\t' + text, file=fp)
