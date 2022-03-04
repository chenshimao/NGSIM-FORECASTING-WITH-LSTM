import torch.nn as nn
import torch


class MyNet(nn.Module):
    def __init__(self, args):
        super(MyNet, self).__init__()
        self.input_size = 4
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.bias = True
        self.batch_first = True
        self.dropout = args.dropout
        self.is_bidirectional = args.is_bidirectional
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            self.num_layers, self.bias, self.batch_first,
                            self.dropout, self.is_bidirectional)
        if self.is_bidirectional:
            self.linear = nn.Linear(in_features=2*self.hidden_size, out_features=1)
        else:
            self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        b, s, h = out.shape
        out = torch.mean(out, dim=1)
        pred = self.linear(out.view(len(x), -1))
        return pred
