import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Preprocces import REL_FEATURES


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # self.lstm = nn.LSTM(len(REL_FEATURES), round(len(REL_FEATURES)/2), num_layers=2, batch_first=True)
        self.hidden2temp = nn.Linear(round(len(REL_FEATURES)), 11)

    def forward(self, x):
        week_size = x.shape[1]
        # print('1', x.shape)
        # lstm_out, _ = self.lstm(x)  # .view(len(sentence), 1, -1)
        # print('2', lstm_out.shape)
        temp_prob = self.hidden2temp(x.view(week_size, -1))
        # print('3', temp_prob.shape)
        return temp_prob
