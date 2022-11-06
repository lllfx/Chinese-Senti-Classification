# -*- coding: UTF-8 -*-
"""
@author:user
@file:rcnn_model.py
@time:2022/11/06
"""
from torch import nn
import torch
import torch.nn.functional as F


class RCNNModel(nn.Module):
    """
     Recurrent Convolutional Neural Networks for Text Classification (2015)
     """

    def __init__(self, vocab_size, num_class, hidden_size=32, hidden_size_linear=64, drop_out=0.5, num_layers=2,
                 padding_idx=0,
                 embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True,
                            dropout=drop_out)
        self.W = nn.Linear(embedding_dim + 2 * hidden_size, hidden_size_linear)
        self.act = nn.Tanh()
        self.fc = nn.Linear(hidden_size_linear, num_class)

    def forward(self, text):
        embed = self.embedding(text)
        output, _ = self.lstm(embed)
        output = torch.cat([output, embed], dim=2)
        output = self.act(self.W(output)).transpose(1, 2)
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        return self.fc(output)


if __name__ == '__main__':
    model = RCNNModel(10, 2)
    print(model.parameters)
    res = model.forward(torch.randint(0, 10, (32, 32)))
    # print(res.shape)
    # print(res)
