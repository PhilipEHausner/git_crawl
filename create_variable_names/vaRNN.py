import torch
import torch.nn as nn


class vaRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size):
        super(vaRNN, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm1 = nn.LSTMCell(embedding_dim * 2, hidden_size)
        self.linear = nn.Linear(hidden_size, num_embeddings)

    def forward(self, sequence, h_t=None, c_t=None):
        output = None
        batch_size, seq_length = sequence.shape

        if not h_t:
            h_t = nn.init.xavier_uniform_(torch.zeros(batch_size, self.hidden_size, dtype=torch.float))
        if not c_t:
            c_t = nn.init.xavier_uniform_(torch.zeros(batch_size, self.hidden_size, dtype=torch.float))
        previous = torch.zeros(batch_size, self.embedding_dim, dtype=torch.float)

        for i in range(seq_length):
            embed = self.embedding(sequence[0, i]).unsqueeze(dim=0)
            embed_and_previous = torch.cat((previous, embed), 1)
            h_t, c_t = self.lstm1(embed_and_previous, (h_t, c_t))
            output = self.linear(h_t)
            previous = self.embedding(torch.argmax(output, dim=1))

        return output

