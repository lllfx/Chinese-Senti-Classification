from torch import nn
import torch


class RnnModel(nn.Module):

    def __init__(self, vocab_size, num_class=2, num_layers=2, hidden_size=32, padding_idx=0, embedding_dim=128,
                 dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, num_class)

    def forward(self, text):
        # print(text.shape)
        embed = self.embedding(text)
        # print(embed.shape)
        # embed = embed.permute(1, 0, 2)
        # print(embed.shape)
        # (seq_len, batch, input_size)
        # output, (h_n, c_n)
        output, _ = self.lstm(embed)
        # output = output.permute(1, 0, 2)
        # print(output.shape)
        # print(output[:, -1, :].shape)
        return self.linear(output[:, -1, :])


if __name__ == '__main__':
    model = RnnModel(10)
    print(model.parameters)
    res = model.forward(torch.LongTensor([[1, 2, 3, 1, 2, 3, 1, 2, 3], [0, 1, 2, 0, 1, 2, 0, 1, 2]]))
    print(res.shape)
    print(res)
