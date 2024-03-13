import json
import pandas as pd
import re
from collections import Counter
from stop_words import get_stop_words
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import math
from collections import OrderedDict

lines = []

with open(r'./eval_output_fixed/eval_predictions.jsonl') as f:
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

def discretize(dictionary):
    oneline = ' '.join(dictionary).lower()
    words = re.findall(r'\b\w+\b', oneline.lower())
    nonstop = filter(lambda w: not w in stopwords, words)
    num_words_overall = len(words)
    return nonstop, num_words_overall

def frequentize(dictionary):
    nonstop, num_words_overall = discretize(dictionary)
    word_frequencies = Counter(nonstop)
    return word_frequencies, num_words_overall

frequentized = {'hypotheses_neutral':{}, 'hypotheses_entail': {}, 'hypotheses_contra': {}}
for key in corresps_standard:
    frequentized[key], num_words_overall[key] = frequentize(corresps_standard[key])

all_standard_premises = frequentized['hypotheses_neutral'] + frequentized['hypotheses_entail'] + frequentized['hypotheses_contra']
result = {'hypotheses_neutral': {}, 'hypotheses_entail': {}, 'hypotheses_contra': {}}
for key in frequentized:
    print(key)
    for word in frequentized[key]:
        # number of times word appears in entailment / number of words overall? in entailment
        p_word_class = frequentized[key][word] / num_words_overall[key]
        # number of times word appears in a hypothesis / number of words
        # print('pwordclass', frequentized[key][word], len(frequentized[key]))
        p_word = all_standard_premises[word] / len(all_standard_premises)
        # print('pword', all_standard_premises[word], len(all_standard_premises))
        # number of hypotheses / number of data lines
        p_class = len(frequentized[key]) / len_standard
        # print('pclass', len(frequentized[key]), len_standard)
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
print(list(adjusted['hypotheses_entail'].items())[0:120])
print(list(adjusted['hypotheses_contra'].items())[0:120])
print(list(adjusted['hypotheses_neutral'].items())[0:120])

# def calc_pmi(hypothesis, dictionary):
#     hyp_pmi = 0
#     for word in nonstop:
#         if word in dictionary:
#             hyp_pmi += dictionary[word]
#     return hyp_pmi    

# calculate pmi for each hypothesis
# for key in corresps_standard:
#     for hyp in corresps_standard[key]:
#         tokenized = re.findall(r'\b\w+\b', hyp.lower())
#         nonstop = filter(lambda w: not w in stopwords, tokenized)
#         calc_pmi(nonstop, adjusted[key])

# combined = list(adjusted['hypotheses_entail'].items()) + list(adjusted['hypotheses_contra'].items()) + list(adjusted['hypotheses_neutral'].items())
# combined = sorted(combined, key=lambda item: item[1])
# combined = [i[0] for i in combined]

# final_set = set()
# for word in combined:
#     final_set.add(word)
#     if len(final_set) == 1024:
#         break

# combined = list(final_set)[0:1024]
# with open("bag.txt", "a") as outfile:
#     for word in combined:
#         outfile.write(word + " ")

json_dict = {}
for key in adjusted:
    adjustedkey = {}
    for word in list(adjusted[key].items()):
        adjustedkey.update({word[0]: word})
    json_dict.update({gold_label_encoding[key]: adjustedkey})

json_obj = json.dumps(json_dict)
with open("fixed_bag.json", "w") as outfile:
    outfile.write(json_obj)