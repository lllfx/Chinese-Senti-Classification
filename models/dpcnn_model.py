from torch import nn
import torch


class DPCNNModel(nn.Module):

    def __init__(self, vocab_size, num_class, max_len, window_size=[2, 3, 4], drop_out=0.5, num_filters=12,
                 padding_idx=0,
                 embedding_dim=128):
        super(DPCNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.conv_region = nn.Conv2d(1, 250,  (3, embedding_dim), stride=1)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        # self.conv = nn.Conv1d(250, 250, kernel_size=3, padding=1)
        # self.pooling = nn.MaxPool1d(3, stride=2)
        # self.act_fun = nn.ReLU()
        # self.linear_out = nn.Linear(2 * self.channel_size, 2)

    def forward(self, x):
        output = self.embedding(x)
        # output = output.permute(0, 2, 1)
        print(output.shape)
        output = output.unsqueeze(1)
        print(output.shape)
        conv_output = self.conv_region(output)
        print(conv_output.shape)
        x = self.padding1(conv_output)  # [batch_size, 250, seq_len, 1]
        print(x.shape)
        # conv_output = self.conv(conv_output)
        # print(conv_output.shape)
        # output = self.act_fun(conv_output)
        # conv_output = self.conv(output)
        # print(conv_output.shape)
        #
        # while conv_output.shape[2] > 2:
        #     pool_out = self.pooling(conv_output)
        #     print(pool_out.shape)
        #     conv_output = self.conv(pool_out)
        #     print(conv_output.shape)
        #     output = self.act_fun(conv_output)
        #     conv_output = self.conv(output)
        #     print(conv_output.shape)
        #     conv_output = conv_output + pool_out


if __name__ == '__main__':
    text_cnn_model = DPCNNModel(100, 2, 9)
    print(text_cnn_model.parameters)
    res = text_cnn_model.forward(torch.LongTensor([[1, 2, 3, 1, 2, 3, 1, 2, 3], [0, 1, 2, 0, 1, 2, 0, 1, 2]]))
    # print(res.shape)
    # print(res)
