import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        
        
        #需要和
        self.conv1 = nn.Conv1d(in_channels=self.args.enc_in,out_channels=64,kernel_size=3,padding=1)
        #self.max_pool1 = nn.MaxPool1d(kernel_size=3)
        self.layer1 = nn.Linear(in_features=64,out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=self.args.c_out)
    
    def forward(self, x):
        # x：[256, 307, 12]
        # print(x.shape)
        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        out = out.permute(0, 2, 1)
        #out = self.max_pool1(out)
        #print("max_pool1", out.shape)
        out = self.layer1(out)
        out = F.relu(out)
        out = self.layer2(out)
        out = F.relu(out)
        # out: [256, 307, 1]
        # out = out.squeeze(dim=2)
        # out: [256, 307]
        
        return out
