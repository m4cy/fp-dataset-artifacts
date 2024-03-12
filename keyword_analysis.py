import json
import pandas as pd
import re
from collections import Counter
from stop_words import get_stop_words
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import math

lines = []

with open(r'./eval_output_premises/eval_predictions.jsonl') as f:
    lines = f.read().splitlines()
len_standard = len(lines)
line_dicts = [json.loads(line) for line in lines]
standard_set = pd.DataFrame(line_dicts)

stopwords=get_stop_words('english')
gold_label_encoding = {'hypotheses_neutral': 1, 'hypotheses_entail': 0, 'hypotheses_contra': 2}
corresps_standard = {}
corresps_standard['hypotheses_neutral'] = standard_set[standard_set['predicted_label'] == 1]['hypothesis']
corresps_standard['hypotheses_entail'] =  standard_set[standard_set['predicted_label'] == 0]['hypothesis']
corresps_standard['hypotheses_contra'] =  standard_set[standard_set['predicted_label'] == 2]['hypothesis']

num_words_overall = {'hypotheses_neutral': 0, 'hypotheses_entail': 0, 'hypotheses_contra': 0}
def frequentize(dictionary):
    oneline = ' '.join(dictionary).lower()
    words = re.findall(r'\b\w+\b', oneline.lower())
    nonstop = filter(lambda w: not w in stopwords, words)
    num_words_overall = len(words)
    word_frequencies = Counter(nonstop)
    return word_frequencies, num_words_overall

for key in corresps_standard:
    corresps_standard[key], num_words_overall[key] = frequentize(corresps_standard[key])

all_standard_premises = corresps_standard['hypotheses_neutral'] + corresps_standard['hypotheses_entail'] + corresps_standard['hypotheses_contra']
result = {'hypotheses_neutral': {}, 'hypotheses_entail': {}, 'hypotheses_contra': {}}
for key in corresps_standard:
    print(key)
    for word in corresps_standard[key]:
        # number of times word appears in entailment / number of words overall? in entailment
        p_word_class = corresps_standard[key][word] / num_words_overall[key]
        # number of times word appears in a hypothesis / number of words
        # print('pwordclass', corresps_standard[key][word], len(corresps_standard[key]))
        p_word = all_standard_premises[word] / len(all_standard_premises)
        # print('pword', all_standard_premises[word], len(all_standard_premises))
        # number of hypotheses / number of data lines
        p_class = len(corresps_standard[key]) / len_standard
        # print('pclass', len(corresps_standard[key]), len_standard)
        # print(word, 'word class', p_word_class, 'pword', p_word, 'pclass', p_class)
        result[key].update({word: abs(math.log((p_word_class) / (p_word * p_class)))})

    result[key] = dict(sorted(result[key].items(), key=lambda item: item[1]))

print(list(result['hypotheses_entail'].items())[0:10])
print(list(result['hypotheses_contra'].items())[0:10])
print(list(result['hypotheses_neutral'].items())[0:10])

adjusted = {'hypotheses_neutral': {}, 'hypotheses_entail': {}, 'hypotheses_contra': {}}
for word in result['hypotheses_contra']:
    n_count = 0
    c_count = 0
    if(word in result['hypotheses_neutral']):
        n_count = result['hypotheses_neutral'][word]
    if(word in result['hypotheses_entail']):
        c_count = result['hypotheses_entail'][word]
    adjusted['hypotheses_contra'][word] = result['hypotheses_contra'][word] - (max(n_count, c_count))

    adjusted['hypotheses_contra'] = dict(sorted(adjusted['hypotheses_contra'].items(), key=lambda item: item[1]))

for word in result['hypotheses_entail']:
    n_count = 0
    c_count = 0
    if(word in result['hypotheses_neutral']):
        n_count = result['hypotheses_neutral'][word]
    if(word in result['hypotheses_contra']):
        c_count = result['hypotheses_contra'][word]
    adjusted['hypotheses_entail'][word] = result['hypotheses_entail'][word] - (max(n_count, c_count))
    adjusted['hypotheses_entail'] = dict(sorted(adjusted['hypotheses_entail'].items(), key=lambda item: item[1]))


for word in result['hypotheses_neutral']:
    n_count = 0
    c_count = 0
    if(word in result['hypotheses_entail']):
        n_count = result['hypotheses_entail'][word]
    if(word in result['hypotheses_contra']):
        c_count = result['hypotheses_contra'][word]
    adjusted['hypotheses_neutral'][word] = result['hypotheses_neutral'][word] - (max(n_count, c_count))
    adjusted['hypotheses_neutral'] = dict(sorted(adjusted['hypotheses_neutral'].items(), key=lambda item: item[1]))


# so what I want is, the words that are most different in each class
print(list(adjusted['hypotheses_entail'].items())[0:30])
print(list(adjusted['hypotheses_contra'].items())[0:30])
print(list(adjusted['hypotheses_neutral'].items())[0:30])



# combined = list(adjusted['hypotheses_entail'].items()) + list(adjusted['hypotheses_contra'].items()) + list(adjusted['hypotheses_neutral'].items())
# combined = sorted(combined, key=lambda item: item[1])
# combined = [i[0] for i in combined][0:1024]
# with open("bag.txt", "a") as outfile:
#     for word in combined:
#         outfile.write(word + " ")

# json_dict = {}
# for key in adjusted:
#     print(len(list(adjusted[key].items())[0:1024]))
#     for word in list(adjusted[key].items())[0:1024]:
#         json_dict.update({word[0]: gold_label_encoding[key]})

# json_obj = json.dumps(json_dict)
# with open("bag.json", "w") as outfile:
#     outfile.write(json_obj)