import collections
import numpy as np

take_duplicates = True

with open('../extract_variable_names/test_vars.txt', 'r') as f:
    real_names = f.readlines()

temp_real_names = []
for i in range(len(real_names)):
    if real_names[i].strip():
        temp_real_names.append(real_names[i].strip())
real_names = temp_real_names

if not take_duplicates:
    real_names = set(real_names)

start_char = collections.Counter([s[0] for s in real_names])
start_char_prob = {k: v/len(real_names) for k, v in start_char.items()}

temp = sorted(list(set(list(''.join(real_names)))))
temp = list(zip(temp, range(len(temp))))
charid = collections.OrderedDict(sorted({item[0]: item[1] for item in temp}.items(), key=lambda t: t[0]))
charid['end'] = 63

bigram_prob = np.zeros([63, 64])
trigram_prob = np.zeros([63, 63, 64])

for name in real_names:
    for i in range(0, len(name)-2):
        bigram_prob[charid[name[i]], charid[name[i+1]]] += 1
        trigram_prob[charid[name[i]], charid[name[i+1]], charid[name[i+2]]] += 1

    if len(name) == 1:
        bigram_prob[charid[name[0]], charid['end']] += 1
    elif len(name) == 2:
        bigram_prob[charid[name[0]], charid[name[1]]] += 1
        bigram_prob[charid[name[1]], charid['end']] += 1
        trigram_prob[charid[name[0]], charid[name[1]], charid['end']] += 1
    else:
        bigram_prob[charid[name[-1]], charid['end']] += 1
        trigram_prob[charid[name[-2]], charid[name[-1]], charid['end']] += 1


bigram_prob = bigram_prob / bigram_prob.sum(axis=1, keepdims=True)
trigram_prob = trigram_prob / (trigram_prob.sum(axis=2, keepdims=True) + np.nextafter(0, 1))


def create_sample(seed=None):
    if seed:
        np.random.seed(seed)

    variable_name = ''

    stop_chance = 0
    curr = np.random.choice(list(start_char_prob.keys()), p=list(start_char_prob.values()))
    variable_name += curr
    curr = np.random.choice(list(charid.keys()), p=bigram_prob[charid[curr], :])
    while curr != 'end':
        variable_name += curr
        if np.random.uniform(0, 1) < stop_chance:
            break
        if trigram_prob[charid[variable_name[-2]], charid[variable_name[-1]], :].sum():
            curr = np.random.choice(list(charid.keys()), p=trigram_prob[charid[variable_name[-2]], charid[variable_name[-1]], :])
        else:
            curr = np.random.choice(list(start_char_prob.keys()), p=list(start_char_prob.values()))
        stop_chance = stop_chance * 2 + 10e-10

    return variable_name
