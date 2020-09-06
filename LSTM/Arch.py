import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Preprocces import REL_FEATURES


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(len(REL_FEATURES), 2*len(REL_FEATURES))
        self.hidden2temp = nn.Linear(2*len(REL_FEATURES), 11)

    def forward(self, x):
        print('1', x)
        lstm_out, _ = self.lstm(x)  # .view(len(sentence), 1, -1)
        print('2', lstm_out)
        tag_space = self.hidden2tag(lstm_out.view(len(x), -1))
        print('3', tag_space)
        tag_scores = F.log_softmax(tag_space, dim=1)
        print('4', tag_scores)
        return
