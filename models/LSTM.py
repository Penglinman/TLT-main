import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args


        self.lstm = nn.LSTM(input_size=args.enc_in, hidden_size=128,num_layers=1,bias=True,batch_first=True,dropout=args.dropout)

        self.fc1 = nn.Linear(in_features=128, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=args.c_out)

    def forward(self, x):

        out, hn = self.lstm(x)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)

        return out