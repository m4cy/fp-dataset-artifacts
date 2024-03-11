import json
import pandas as pd
import re
from collections import Counter
from stop_words import get_stop_words
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

lines = []
with open(r'./eval_output_contrast/eval_predictions.jsonl') as f:
    lines = f.read().splitlines()
len_contrast = len(lines)
line_dicts = [json.loads(line) for line in lines]
contrast_set = pd.DataFrame(line_dicts)
with open(r'./eval_output_standard/eval_predictions.jsonl') as f:
    lines = f.read().splitlines()
len_standard = len(lines)
line_dicts = [json.loads(line) for line in lines]
standard_set = pd.DataFrame(line_dicts)

stopwords=get_stop_words('english')

corresps_contrast = {}
corresps_contrast['hypotheses_neutral'] = contrast_set[contrast_set['label'] == 1]['hypothesis']
corresps_contrast['hypotheses_entail'] =  contrast_set[contrast_set['label'] == 0]['hypothesis']
corresps_contrast['hypotheses_contra'] =  contrast_set[contrast_set['label'] == 2]['hypothesis']

corresps_standard = {}
corresps_standard['hypotheses_neutral'] = standard_set[standard_set['label'] == 1]['hypothesis']
corresps_standard['hypotheses_entail'] =  standard_set[standard_set['label'] == 0]['hypothesis']
corresps_standard['hypotheses_contra'] =  standard_set[standard_set['label'] == 2]['hypothesis']


def frequentize(dictionary):
    oneline = ' '.join(dictionary).lower()
    words = re.findall(r'\b\w+\b', oneline.lower())
    nonstop = filter(lambda w: not w in stopwords, words)
    word_frequencies = Counter(nonstop)
    return word_frequencies

# normalize frequencies
# not handling hypothesis yet so this will be slightly off for those
def normalize_dict(dictionary, num_tokens):
    for key in dictionary:
        for item in dictionary[key]:
            dictionary[key][item] /= num_tokens
            dictionary[key][item] *= 100000
    return dictionary


for key in corresps_contrast:
    corresps_contrast[key] = frequentize(corresps_contrast[key])

all_contrast_premises = corresps_contrast['hypotheses_neutral'] + corresps_contrast['hypotheses_entail'] + corresps_contrast['hypotheses_contra']

for key in corresps_standard:
    corresps_standard[key] = frequentize(corresps_standard[key])

all_standard_premises = corresps_standard['hypotheses_neutral'] + corresps_standard['hypotheses_entail'] + corresps_standard['hypotheses_contra']

corresps_standard = normalize_dict(corresps_standard, len(all_standard_premises))
corresps_contrast = normalize_dict(corresps_contrast, len(all_contrast_premises))

result = {}
for key in corresps_standard:
    result[key] = corresps_standard[key] - corresps_contrast[key]

# okay now redo the pipeline all of that over but on false positives and false negatives?
# should I group all premises or keep them distinct based on entailment, neutrality, and contradiction?
mislabeled_contrast = {}
mislabeled_contrast_lines = contrast_set[contrast_set['label'] != contrast_set['predicted_label']]
mislabeled_contrast['hypotheses_neutral'] = mislabeled_contrast_lines[mislabeled_contrast_lines['label'] == 1]['hypothesis']
mislabeled_contrast['hypotheses_entail'] = mislabeled_contrast_lines[mislabeled_contrast_lines['label'] == 0]['hypothesis']
mislabeled_contrast['hypotheses_contra'] = mislabeled_contrast_lines[mislabeled_contrast_lines['label'] == 2]['hypothesis']
mislabeled_standard = {}
mislabeled_standard_lines = standard_set[standard_set['label'] != standard_set['predicted_label']]
mislabeled_standard['hypotheses_neutral'] = mislabeled_standard_lines[mislabeled_standard_lines['label'] == 1]['hypothesis']
mislabeled_standard['hypotheses_entail'] = mislabeled_standard_lines[mislabeled_standard_lines['label'] == 0]['hypothesis']
mislabeled_standard['hypotheses_contra'] = mislabeled_standard_lines[mislabeled_standard_lines['label'] == 2]['hypothesis']
for key in mislabeled_contrast:
    mislabeled_contrast[key] = frequentize(mislabeled_contrast[key])

for key in mislabeled_standard:
    mislabeled_standard[key] = frequentize(mislabeled_standard[key])

all_mislabeled_premises_contrast = mislabeled_contrast['hypotheses_neutral'] + mislabeled_contrast['hypotheses_entail'] + mislabeled_contrast['hypotheses_contra']
all_mislabeled_premises_standard = mislabeled_standard['hypotheses_neutral'] + mislabeled_standard['hypotheses_entail'] + mislabeled_standard['hypotheses_contra']
mislabeled_contrast = normalize_dict(mislabeled_contrast, len(all_mislabeled_premises_contrast))
mislabeled_standard = normalize_dict(mislabeled_standard, len(all_mislabeled_premises_standard))
# print(mislabeled_contrast)
# print(mislabeled_standard)
mis_result = {}
for key in mislabeled_standard:
    mis_result[key] = mislabeled_standard[key] - mislabeled_contrast[key]
    print(key)
    print(mis_result[key].most_common(5))
y_true = contrast_set['label']
y_pred = contrast_set['predicted_label']
contrast_results = metrics.classification_report(y_true, y_pred)
cm = metrics.confusion_matrix(y_true, y_pred)
print(contrast_results)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('./beep.jpg')

y_true = standard_set['label']
y_pred = standard_set['predicted_label']
standard_results = metrics.classification_report(y_true, y_pred)
cm_2 = metrics.confusion_matrix(y_true, y_pred)
print(standard_results)
sns.heatmap(cm_2, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('./meep.jpg')
