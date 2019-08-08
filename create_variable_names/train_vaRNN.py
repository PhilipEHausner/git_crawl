import torch
from torch import nn
import collections
import numpy as np
from vaRNN import vaRNN


def one_hot(x, length):
    t = torch.zeros(length)
    t[x] = 1
    return t

take_duplicates = False

with open('../extract_variable_names/test_vars.txt', 'r') as f:
    real_names = f.readlines()

temp_real_names = []
for i in range(min(1000, len(real_names))):
    if real_names[i].strip():
        temp_real_names.append(real_names[i].strip())
real_names = temp_real_names

if not take_duplicates:
    real_names = list(set(real_names))

start_char = collections.Counter([s[0] for s in real_names])
start_char_prob = {k: v/len(real_names) for k, v in start_char.items()}

temp = sorted(list(set(list(''.join(real_names)))))
temp = list(zip(temp, range(len(temp))))
charid = collections.OrderedDict(sorted({item[0]: item[1] for item in temp}.items(), key=lambda t: t[0]))
charid['end'] = 63
charid['start'] = 64

np.random.shuffle(real_names)

x_train = real_names[:int(len(real_names) * 0.8)]
x_test = real_names[int(len(real_names) * 0.8):]
assert len(x_train) + len(x_test) == len(real_names)

x_train_flattened = []
y_train_flattened = []
for i, item in enumerate(x_train):
    whole_word = [item for sublist in [[64], [charid[l] for l in list(item)], [63]] for item in sublist]
    for j in range(1, len(whole_word)):
        x_train_flattened.append(torch.tensor(whole_word[:j], dtype=torch.long))
        y_train_flattened.append(torch.tensor(whole_word[j], dtype=torch.long))
        # y_train_flattened.append(torch.eye(65, dtype=torch.long)[whole_word[j], :])

x_test_flattened = []
y_test_flattened = []
for i, item in enumerate(x_test):
    whole_word = [item for sublist in [[64], [charid[l] for l in list(item)], [63]] for item in sublist]
    for j in range(1, len(whole_word)):
        x_test_flattened.append(torch.tensor(whole_word[:j], dtype=torch.long))
        y_test_flattened.append(torch.eye(65, dtype=torch.long)[whole_word[j], :])


epochs = 20
net = vaRNN(65, 8, 32)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
print('Start training.')
for i in range(epochs):
    losses = []
    for j, sample in enumerate(x_train_flattened):
        optimizer.zero_grad()
        prediction = net(sample.unsqueeze(dim=0))
        curr_loss = loss(prediction, y_train_flattened[j].unsqueeze(dim=0))
        losses.append(curr_loss.item())
        curr_loss.backward()
        optimizer.step()
    print('Epoch', i+1, ':', np.mean(losses))

# output = 0
# t = torch.tensor([64], dtype=torch.long).unsqueeze(dim=0)
# while output != 63:
#     output = net(t)
#     output = torch.argmax(output, dim=1)
#     t = torch.cat((t, torch.tensor([output], dtype=torch.long).unsqueeze(dim=0)), 1)
#
# rev = {value: key for key, value in charid.items()}
# ''.join([rev[x.item()] for x in t[0] if x != 63 and x != 64])
