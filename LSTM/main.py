import torch
import torch.nn as nn
from torch import optim

from Arch import Net
from Preprocces import train_loader, test_loader
from sklearn.metrics import accuracy_score


class Main:
    def __init__(self):
        self.batch_size = 10
        self.model = Net()

    def evaluate(self):
        trues = []
        preds = []
        loss_sum = 0
        loss_function = nn.MSELoss()
        # loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(train_loader):
                if y.shape[1] == 1:
                    continue

                p = self.model(x)

                p = torch.argmax(p, dim=1).type(torch.float32)
                y = torch.squeeze(y.type(torch.float32))

                loss_sum += loss_function(p, y) / self.batch_size

                trues += y
                preds += p
        print('loss =', loss_sum)
        print(trues[:5])
        print(preds[:5])
        print('accu =', accuracy_score(trues, preds))

    def train(self):
        # loss_function = nn.NLLLoss()
        # loss_function = nn.CrossEntropyLoss()
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())
        self.model.zero_grad()

        for epoch in range(80):
            los_sum = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                if y.shape[1] == 1:
                    continue

                self.model.zero_grad()
                if batch_idx == 0:
                    print(x.shape)
                    print(x[:, :3, :])
                p = self.model(x)
                # p.requires_grad = True
                # p = torch.tensor(p, requires_grad=True)

                # MSE
                p = torch.argmax(p, dim=1).type(torch.float32).requires_grad_(True)
                y = torch.squeeze(y.type(torch.float32))

                # Cross-Entropy / NLLLoss
                # y = y.view(y.shape[1])

                loss = loss_function(p, y) / self.batch_size  # calculate the loss
                # break
                # print(self.model.hidden2temp.weight)
                loss.backward()
                optimizer.step()
                # print(self.model.hidden2temp.weight)

                los_sum += loss.item()

                # if batch_idx % 600 == 0:
                # print('data')
                # print(x[:5])
                # print(self.model.hidden2temp.weight.grad)
                # print(loss.data)
                # if batch_idx % self.batch_size == 0 and batch_idx != 0:  # make a step
                #     optimizer.step()
                #     self.model.zero_grad()

            if epoch % 1 == 0:
                print(epoch, 'train loss', los_sum)

                self.evaluate()


if __name__ == '__main__':
    Main().train()
