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
        # loss_function = nn.MSELoss()
        loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                if y.shape[1] == 1:
                    continue
                p = self.model(x)
                # print(torch.argmax(p, dim=1).shape)
                # print(torch.squeeze(y).shape)
                loss_sum += loss_function(p, torch.squeeze(y)) #/ self.batch_size  # calculate the loss
                # loss_sum += loss_function(torch.argmax(p, dim=1), torch.squeeze(y.type(torch.float32))) / self.batch_size  # calculate the loss
                trues += torch.squeeze(y)
                preds += torch.argmax(p, dim=1)
        print('loss =', loss_sum)
        print('accu =', accuracy_score(trues, preds))

    def train(self):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        self.model.zero_grad()

        for epoch in range(8):
            los_sum = 0
            print(len(train_loader))
            for batch_idx, (x, y) in enumerate(train_loader):
                if y.shape[1] == 1:
                    continue
                p = self.model(x)

                loss = loss_function(p, torch.squeeze(y)) #/ self.batch_size  # calculate the loss
                loss.backward()

                los_sum += loss

                if batch_idx % 100 == 0:
                    # print(self.model.hidden2temp.weight.grad)
                    print(loss.data)
                if batch_idx % self.batch_size == 0 and batch_idx != 0:  # make a step
                    optimizer.step()
                    self.model.zero_grad()

            if epoch % 1 == 0:
                print(epoch, 'loss', los_sum)

                self.evaluate()


if __name__ == '__main__':
    Main().train()
