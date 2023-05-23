import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(args.enc_in, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, args.c_out),
        )

        # self.layer1 = nn.Linear(in_features=args.seq_len, out_features=5)
        # self.layer2 = nn.Linear(in_features=32, out_features=16)
        # self.layer3 = nn.Linear(in_features=5, out_features=1)
        #self.layer4 = nn.Linear(in_features=512, out_features=1)

        #self.layer5 = nn.Linear(in_features=16, out_features=1)
        #self.layer6 = nn.Linear(in_features=2, out_features=1)
    
    def forward(self, x):
        # x：[256, 307, 12]
        out = self.MLP(x)
        # out: [256, 307, 128]
        out = F.relu(out)
        #out = self.layer2(out)
        #out = F.relu(out)
        # out = self.layer3(out)
        # out = F.relu(out)
        #out = self.layer4(out)
        #out = F.relu(out)
        #out = self.layer5(out)
        #out = F.relu(out)
        #out = self.layer6(out)
        #out = F.relu(out)
        # out: [256, 307, 1]
        # out = out.squeeze(dim=2)
        # out: [256, 307]
        
        return out
