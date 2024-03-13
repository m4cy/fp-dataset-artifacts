import pandas as pd
import json

with open('fixed_bag.json') as f:
    fixed_set = json.load(f)


with open('original_bag.json') as f:
    original_set = json.load(f)

diff_set = {'0': {}, '1': {}, '2': {}}
for key in original_set:
    for word in original_set[key]:
        diff = 0
        if word in fixed_set[key]: 
            diff = fixed_set[key][word][1]
        diff_set[key][word] = original_set[key][word][1] - diff

for key in diff_set:
    diff_set[key] = dict(sorted(diff_set[key].items(), key=lambda item: item[1]))

print(list(diff_set['0'].items())[0:30])
print(list(diff_set['1'].items())[0:30])
print(list(diff_set['2'].items())[0:30])

