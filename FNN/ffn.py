import torch
import torch.nn as nn
from torch import optim

from Arch import Net
from Preprocces import train_loader, test_loader
from sklearn.metrics import precision_recall_fscore_support


def evaluate():
    trues = []
    preds = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x = batch[0]
            y = batch[1]
            p = model(x)
            l = torch.argmax(p, dim=1)
            trues += y
            preds += l
    # print(acc)
    print(precision_recall_fscore_support(trues, preds, average='binary'))


model = Net()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(8):
    for batch_idx, batch in enumerate(train_loader):
        x = batch[0]
        y = batch[1]
        p = model(x)
        loss = loss_function(p, y)
        loss.backward()
        optimizer.step()
        model.zero_grad()

    if epoch % 1 == 0:
        print(epoch)
        evaluate()
