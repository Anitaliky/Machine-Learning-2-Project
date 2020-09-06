import torch
import torch.nn as nn
from torch import optim

from Arch import Net
from Preprocces import train_loader, test_loader
from sklearn.metrics import accuracy_score

class Main():
    def __init__(self):
        self.batch_size = 10
        self.model = Net()

    def evaluate(self):
        trues = []
        preds = []
        loss_sum = 0
        loss_function = nn.MSELoss()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                if y.shape[1] == 1:
                    continue
                p = self.model(x)
                # print(torch.argmax(p, dim=1).shape)
                # print(torch.squeeze(y).shape)
                loss_sum += loss_function(torch.argmax(p, dim=1), torch.squeeze(y.type(torch.float32))) / self.batch_size  # calculate the loss
                trues += torch.squeeze(y)
                preds += torch.argmax(p, dim=1)
        print(loss_sum)
        print(accuracy_score(trues, preds))

    def train(self):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.model.zero_grad()
        for epoch in range(8):
            for batch_idx, (x, y) in enumerate(train_loader):
                if y.shape[1] == 1:
                    continue
                p = self.model(x)
                # print(torch.squeeze(y).shape)
                loss = loss_function(p, torch.squeeze(y)) / self.batch_size  # calculate the loss
                loss.backward()
                if batch_idx % 100 == 0:
                    print(self.model.hidden2temp.weight.grad)
                    print(loss.data)
                if batch_idx % self.batch_size == 0 and batch_idx != 0:  # make a step
                    optimizer.step()
                    self.model.zero_grad()
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            print(name, param.grad, end=', ')
                        else:
                            print()
                            print(name)
                    print()
                        # break

            if epoch % 1 == 0:
                print(epoch)
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(name, param.data)
                        break

                self.evaluate()


if __name__ == '__main__':
    Main().train()
