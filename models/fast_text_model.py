from torch import nn
import torch
import torch.nn.functional as F


class FastTextModel(nn.Module):

    def __init__(self, vocab_size, num_classes, n_gram_vocab, dropout=0.5, padding_idx=0, embedding_dim=128,
                 hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_ngram2 = nn.Embedding(n_gram_vocab, embedding_dim)
        self.embedding_ngram3 = nn.Embedding(n_gram_vocab, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedding_list = []
        for text, emb in zip([x[0], x[1], x[2]], [self.embedding, self.embedding_ngram2, self.embedding_ngram3]):
            embedding_list.append(emb(text))
        embedding = torch.cat(embedding_list, dim=-1)
        embedding_mean = embedding.mean(dim=1)
        embedding_mean_dropout = self.dropout(embedding_mean)
        fc1_out = self.fc1(embedding_mean_dropout)
        fc1_out = F.relu(fc1_out)
        fc2_out = self.fc2(fc1_out)
        return fc2_out
