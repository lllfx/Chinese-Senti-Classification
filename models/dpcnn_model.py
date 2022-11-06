from torch import nn
import torch
import torch.nn.functional as F


class DPCNNModel(nn.Module):

    def __init__(self, vocab_size, num_class, drop_out=0.5, num_filters=12, padding_idx=0, embedding_dim=128):
        super(DPCNNModel, self).__init__()
        self.num_filters = num_filters
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.conv_region = nn.Conv2d(1, self.num_filters, (3, embedding_dim), stride=1)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (3, 1), stride=1)
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2 * self.num_filters, num_class)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        batch = x.shape[0]
        output = self.embedding(x)
        output = self.dropout(output)
        output = output.unsqueeze(1)
        conv_output = self.conv_region(output)
        padding_out = self.padding_conv(conv_output)  # [batch_size, 250, seq_len, 1]
        padding_out = self.act_fun(padding_out)
        conv_output = self.conv(padding_out)
        padding_out = self.padding_conv(conv_output)  # [batch_size, 250, seq_len, 1]
        padding_out = self.act_fun(padding_out)
        x = self.conv(padding_out)
        # 开始迭代
        while x.shape[2] > 2:
            x = self._block(x)
        # 判断
        if x.shape[2] == 1:
            x = x.squeeze(2)
            x = torch.stack([x, x], dim=2)
            x = x.view(batch, 2 * self.num_filters)
        else:
            x = x.view(batch, 2 * self.num_filters)

        x = self.linear_out(x)
        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)
        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv(x)
        # Short Cut
        x = x + px
        return x


if __name__ == '__main__':
    text_cnn_model = DPCNNModel(100, 2, 9)
    print(text_cnn_model.parameters)
    res = text_cnn_model.forward(torch.randint(0, 100, (32, 32)))
    # print(res.shape)
    # print(res)
