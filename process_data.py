import pandas as pd

d = pd.read_csv('./snli_1.0_dev.txt', sep='\t')

d = d[['sentence1','sentence2', 'gold_label']]
d.dropna(inplace=True)
gold_label_encoding = {'neutral': 1, 'entailment': 0, 'contradiction': 2}
d.drop(d[d['gold_label'] == '-'].index, inplace=True)
d['gold_label'] = [gold_label_encoding[value] for value in d['gold_label']]
d = d.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis', 'gold_label': 'label'})
d.to_csv('snli_dev.txt', sep = '\t')