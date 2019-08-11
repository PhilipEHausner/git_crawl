import torch
from torch import nn
import collections
import numpy as np
from vaRNN import vaRNN
from torch.nn.utils.rnn import pad_sequence
import timeit

def validate(test_x, test_y, network, losss):
    network.eval()
    lossses = []
    for j, sample_probam in enumerate(test_x):
        prediction_probam = net(sample_probam.unsqueeze(dim=0))
        curr_loss_probam = losss(prediction_probam, test_y[j].unsqueeze(dim=0))
        lossses.append(curr_loss_probam.item())
    network.train()
    return np.mean(lossses)


def eval_net(start):
    start = list(start)
    start = [charid[s] for s in start]
    start = [64] + start
    print(start)
    output = 0
    t = torch.tensor(start, dtype=torch.long).unsqueeze(dim=0)
    while output != 63 and t.shape[1] < 32:
        output = net(t)
        output = torch.argmax(output, dim=1)
        t = torch.cat((t, torch.tensor([output], dtype=torch.long).unsqueeze(dim=0)), 1)

    rev = {value: key for key, value in charid.items()}
    return ''.join([rev[x.item()] for x in t[0] if x != 63 and x != 64])


if __name__ == "__main__":
    take_duplicates = False

    with open('../extract_variable_names/test_vars.txt', 'r') as f:
        real_names = f.readlines()

    temp_real_names = []
    for i in range(min(100000, len(real_names))):
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
            y_test_flattened.append(torch.tensor(whole_word[j], dtype=torch.long))

    epochs = 20
    net = vaRNN(65, 8, 16, 65)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    prev_best = (0, np.inf)
    batch_size = 4
    print('Start training.')
    for i in range(epochs):
        start = timeit.default_timer()
        losses = []

        for j in range(0, len(x_train_flattened), batch_size):
            batch_x = x_train_flattened[j:j+batch_size]
            batch_y = torch.stack(y_train_flattened[j:j+batch_size])
            batch_x = pad_sequence(batch_x, batch_first=True, padding_value=65)
            optimizer.zero_grad()
            prediction = net(batch_x)
            curr_loss = loss(prediction, batch_y)
            losses.append(curr_loss.item())
            curr_loss.backward()
            optimizer.step()
            batch = []

        test_result = validate(x_test_flattened, y_test_flattened, net, loss)
        stop = timeit.default_timer()
        print('Epoch', i+1, ':', np.round(np.mean(losses), 5), '(', round(stop - start), 'seconds ),  validation loss:',
              np.round(test_result, 5))
        if test_result < prev_best[1]:
            torch.save(net, './lstm_curr_best.pt')
            prev_best = (i, test_result)
        else:
            if i - prev_best[0] > 2:
                break





