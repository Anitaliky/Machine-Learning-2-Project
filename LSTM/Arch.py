import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Preprocces import REL_FEATURES


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(len(REL_FEATURES), len(REL_FEATURES), num_layers=2)
        self.hidden2temp = nn.Linear(len(REL_FEATURES), 11)

    def forward(self, x):
        # print('1', x.shape)
        lstm_out, _ = self.lstm(x)  # .view(len(sentence), 1, -1)
        # print('2', lstm_out.shape)
        temp_prob = self.hidden2temp(lstm_out[0])
        # print('3', temp_prob.shape)
        return temp_prob
