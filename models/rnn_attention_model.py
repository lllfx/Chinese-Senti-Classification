from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

NEG_INF = -10000
TINY_FLOAT = 1e-6


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)
    return mask


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, x, seq_lens):
        # # sort input by descending length
        # _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        # _, idx_unsort = torch.sort(idx_sort, dim=0)
        # x_sort = torch.index_select(x, dim=0, index=idx_sort)
        # seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)
        # # pack input
        # x_packed = pack_padded_sequence(
        #     x_sort, seq_lens_sort, batch_first=True,enforce_sorted=False)
        # print(x.shape, seq_lens)
        x_packed = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        y_packed, _ = self.lstm(x_packed)
        unpacked_seq, unpacked_lens = pad_packed_sequence(y_packed, batch_first=True)
        # print(seq_lens)
        # print(unpacked_lens)
        # print(unpacked_seq.shape)
        # # pass through rnn
        return unpacked_seq


def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result


def mask_mean(seq, mask=None):
    """Compute mask average on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_mean : torch.float, size [batch, n_channels]
        Mask mean of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    mask_sum = torch.sum(  # [b,msl,nc]->[b,nc]
        seq * mask.unsqueeze(-1).float(), dim=1)
    seq_len = torch.sum(mask, dim=-1)  # [b]
    mask_mean = mask_sum / (seq_len.unsqueeze(-1).float() + TINY_FLOAT)

    return mask_mean


def mask_max(seq, mask=None):
    """Compute mask max on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_max : torch.float, size [batch, n_channels]
        Mask max of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)
    mask_max, _ = torch.max(  # [b,msl,nc]->[b,nc]
        seq + (1 - mask.unsqueeze(-1).float()) * NEG_INF, dim=1)

    return mask_max


class RnnAttentionModel(nn.Module):

    def __init__(self, vocab_size, num_class=2, num_layers=2, hidden_size=32, padding_idx=0, embedding_dim=128,
                 dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = DynamicLSTM(embedding_dim, hidden_size, num_layers, dropout, bidirectional=True)
        self.fc_att = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 6, hidden_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, num_class)

    def forward(self, text, seq_lens):
        # print(text.shape)
        embed = self.embedding(text)
        # print(embed.shape)
        # embed = embed.permute(1, 0, 2)
        # print(embed.shape)
        # (seq_len, batch, input_size)
        # output, (h_n, c_n)
        output = self.lstm(embed, seq_lens)
        # print(output.shape)
        att = self.fc_att(output).squeeze(-1)
        # print(att.shape)

        max_seq_len = torch.max(seq_lens).item()
        mask = seq_mask(seq_lens, max_seq_len)  # [b,msl]
        # print(max_seq_len)
        # print(mask)
        att = mask_softmax(att, mask)  # [b,msl]
        # print(att.shape)
        # print(att)
        output_att = torch.sum(att.unsqueeze(-1) * output, dim=1)
        # print(output_att.shape)

        # pooling
        output_avg = mask_mean(output, mask)  # [b,h*2]
        output_max = mask_max(output, mask)  # [b,h*2]
        output = torch.cat([output_avg, output_max, output_att], dim=-1)  # [b,h*6]
        # print(output.shape)
        # feed-forward
        f = self.drop(self.act(self.fc(output)))  # [b,h*6]->[b,h]
        return self.out(f)


if __name__ == '__main__':
    model = RnnAttentionModel(10)
    print(model.parameters)
    x = torch.LongTensor(
        [[1, 2, 3, 1, 2, 0, 0, 0, 0, 0], [1, 2, 3, 1, 2, 2, 3, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]])
    seq_lens = torch.LongTensor([5, 7, 3])
    res = model.forward(x, seq_lens)
    print(res.shape)
    print(res)

# model = DynamicLSTM(10)
# print(model.parameters)
# seq_lens = torch.LongTensor([5, 2, 3, 4, 1])
# print(seq_lens)
# _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
# print(idx_sort)
# _, idx_unsort = torch.sort(idx_sort, dim=0)
# print(idx_unsort)
