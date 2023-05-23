import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        self.conv1 = nn.Conv1d(in_channels=self.args.enc_in, out_channels=64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128,num_layers=1,bias=True,batch_first=True,dropout=args.dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.attention = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(in_features=128, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=args.c_out)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        out = out.permute(0, 2, 1)
        out, hn = self.lstm(out)
        out = self.attention(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)

        return out