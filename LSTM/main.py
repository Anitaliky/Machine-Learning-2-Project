import torch
import torch.nn as nn
from torch import optim

from Arch import Net
from Preprocces import train_loader, test_loader, REL_FEATURES
from sklearn.metrics import accuracy_score


class Main:
    def __init__(self, model_name):
        self.model_name = model_name
        self.batch_size = 10
        self.model = Net()

    def evaluate(self):
        trues = []
        preds = []
        loss_sum = 0
        loss_function = nn.MSELoss()
        # loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            last_group = {'year': None, 'month': None}
            for batch_idx, (x, y) in enumerate(train_loader):
                if y.shape[1] == 1:
                    print('continued')
                    continue

                p = self.model(x).round()
                y = y.view(p.shape)

                curr_group = {name: x[0][-1][REL_FEATURES.index(name)] for name in ['year', 'month']}
                if not (batch_idx == 0 or
                        ('full' not in self.model_name and 'winter' not in self.model_name and
                         last_group['year'] != curr_group['year']) or
                        ('winter' in self.model_name and curr_group['month'] != '1')):
                    y = y[2:]
                    p = p[2:]

                loss_sum += loss_function(p, y) / self.batch_size

                trues += y
                preds += p
                last_group = curr_group
        print('test loss =', loss_sum)
        # print(trues[:5])
        # print(preds[:5])
        print('accu =', accuracy_score(trues, preds))

    def train(self):
        # loss_function = nn.NLLLoss()
        # loss_function = nn.CrossEntropyLoss()
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        self.model.zero_grad()

        for epoch in range(800):
            los_sum = 0
            last_group = {'year': None, 'month': None}
            for batch_idx, (x, y) in enumerate(train_loader):
                if y.shape[1] == 1:
                    print('continued')
                    continue

                self.model.zero_grad()
                p = self.model(x)
                y = y.view(p.shape)
                # p.requires_grad = True
                # p = torch.tensor(p, requires_grad=True)
                curr_group = {name: x[0][-1][REL_FEATURES.index(name)] for name in ['year', 'month']}
                if not (batch_idx == 0 or \
                        ('full' not in self.model_name and 'winter' not in self.model_name and
                         last_group['year'] != curr_group['year']) or \
                        ('winter' in self.model_name and curr_group['month'] != '1')):
                    y = y[2:]
                    p = p[2:]
                loss = loss_function(p, y) / self.batch_size  # calculate the loss

                if (batch_idx+1) % self.batch_size == 0:
                    loss.backward()
                    optimizer.step()

                los_sum += loss.item()
                last_group = curr_group

            if epoch % 1 == 0:
                print('\n', epoch, 'train loss', los_sum)

                self.evaluate()


if __name__ == '__main__':
    Main('full_all').train()
    # for freq_range in [[str(i) for i in range(1, 13)],  # month
    #                    ['autumn', 'winter', 'spring', 'monsoon', 'summer'],  # season
    #                    ['full']]:  # full year
    #     for freq_part in freq_range:
    #         for day_part in ['all', 'night', 'morning', 'noon', 'evening']:
    #             print('\n')
    #             print('_'.join([freq_part, day_part]))
