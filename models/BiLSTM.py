import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()


        self.lstm = nn.LSTM(input_size=args.enc_in, hidden_size=128,num_layers=1,bias=True,batch_first=True,dropout=args.dropout, bidirectional = True)

        self.MLP = nn.Sequential(
            nn.Linear(256, 256),  # BiLSTM隐藏层输出维度是hidden_size*2
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, args.c_out),
        )


    def forward(self, x):

        out, hn = self.lstm(x)
        out = self.MLP(out)
        out = F.relu(out)

        return out