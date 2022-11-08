import torch
import torch.nn as nn


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    """input_size:输入 xt 的特征维度
       hidden_size：表示输出的特征维度
       num_layers：表示网络的层数
       nonlinearity：表示选用的非线性激活函数，默认是 ‘tanh’
       bias：表示是否使用偏置，默认使用
       batch_first：表示输入数据的形式，默认是 False，就是这样形式，(seq, batch, feature)，也就是将序列长度放在第一位，batch 放在第二位
       dropout：表示是否在输出层应用 dropout
       bidirectional：表示是否使用双向的 LSTM，默认是 False。
    """

    def __init__(self, embedding_dim, hidden_size, num_classes, num_layers, bidirectional):
        super(LSTM, self).__init__()
        # self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=word2idx['<PAD>'])
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        # seq_len = x.shape[2]
        # 初始化一个h0,也即c0，在RNN中一个Cell输出的ht和Ct是相同的，而LSTM的一个cell输出的ht和Ct是不同的
        # 维度[layers, batch, hidden_len]
        if self.bidirectional:
            h0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).cuda()
            c0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).cuda()
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
        # x = self.embedding(x)
        out, (_, _) = self.lstm(x, (h0, c0))
        output = self.fc(out[:, -1, :]).squeeze(0)  # 因为有max_seq_len个时态，所以取最后一个时态即-1层
        return output
