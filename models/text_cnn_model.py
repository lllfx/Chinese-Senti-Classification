from torch import nn
import torch


class TextCnnModel(nn.Module):

    def __init__(self, vocab_size, num_class, max_len, window_size=[2, 3, 4], drop_out=0.5, num_filters=12,
                 padding_idx=0,
                 embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.drop_out = nn.Dropout(drop_out)
        self.conv_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_filters, kernel_size=h),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=max_len - h + 1)
            ) for h in window_size
        ])
        self.fc = nn.Linear(in_features=num_filters * len(window_size),
                            out_features=num_class)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.drop_out(emb)
        emb = emb.permute(0, 2, 1)
        output_list = [conv(emb) for conv in self.conv_list]
        output = torch.cat(output_list, dim=1)
        output = output.view(-1, output.size(1))
        return self.fc(output)


if __name__ == '__main__':
    # m = nn.Conv1d(16, 33, 3, stride=2)
    # print(m)
    # input = torch.randn(20, 16, 50)
    # print(input.shape)
    # output = m(input)
    # print(output.shape)
    text_cnn_model = TextCnnModel(100, 2, 9)
    print(text_cnn_model.parameters)
    res = text_cnn_model.forward(torch.LongTensor([[1, 2, 3, 1, 2, 3, 1, 2, 3], [0, 1, 2, 0, 1, 2, 0, 1, 2]]))
    print(res.shape)
    print(res)
