import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Preprocces import REL_FEATURES


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(len(REL_FEATURES), round(len(REL_FEATURES)/2), num_layers=2, batch_first=True)
        self.hidden2bars = nn.Linear(round(len(REL_FEATURES)/2), 1)
        # self.bars2bar = nn.Linear(11, 1)

    def forward(self, x):
        week_size = x.shape[1]
        # print('1', x.shape)
        x, _ = self.lstm(x)  # .view(len(sentence), 1, -1)
        # print('2', x.shape)
        x = F.relu(x)
        x = self.hidden2bars(x.view(week_size, -1))
        return x
        # print('3', x.shape)
        x = self.bars2bar(x.view(week_size, -1))
        return x
