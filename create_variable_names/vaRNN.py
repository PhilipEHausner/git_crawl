import torch
import torch.nn as nn


class vaRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, padding_idx):
        super(vaRNN, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        # Make vocabulary one element bigger for padding index
        self.embedding = nn.Embedding(num_embeddings+1, embedding_dim, padding_idx=padding_idx)
        self.lstm1 = nn.LSTMCell(embedding_dim * 2, hidden_size)
        self.linear = nn.Linear(hidden_size, num_embeddings)

    def forward(self, sequence, h_t=None, c_t=None):
        output = None
        batch_size, seq_length = sequence.shape

        if not h_t:
            h_t = nn.init.xavier_uniform_(torch.zeros(batch_size, self.hidden_size, dtype=torch.float))
        if not c_t:
            c_t = nn.init.xavier_uniform_(torch.zeros(batch_size, self.hidden_size, dtype=torch.float))

        # batch_size, embedding_dim
        previous = torch.zeros(batch_size, self.embedding_dim, dtype=torch.float)

        # batch_size, sequence_length, embedding_dim
        embed = self.embedding(sequence)

        for i in range(seq_length):
            # batch_size, embedding_dim * 2
            embed_and_previous = torch.cat((previous, embed[:, i, :].squeeze(dim=1)), 1)

            # batch_size, hidden_size; batch_size, hidden_size
            h_t, c_t = self.lstm1(embed_and_previous, (h_t, c_t))
            # batch_size, num_embeddings
            output = self.linear(h_t)
            # batch_size, 1, embedding_dim
            previous = self.embedding(torch.argmax(output, dim=1))

        return output

