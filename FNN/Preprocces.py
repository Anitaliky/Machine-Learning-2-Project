import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
import pandas as pd

REL_FEATURES = ['Stop', 'Station', 'Visibility(mi)', 'Junction', 'Crossing',
       'Wind_Speed(mph)', 'Pressure(in)', 'Humidity(%)', 'Traffic_Signal',
       'Temperature(F)', 'Distance(mi)', 'Start_Lat', 'Start_Lng']

df = pd.read_csv('train.csv')
df = df[REL_FEATURES]

data_tensor = torch.tensor(df.drop('Severity', axis=1).values.astype('float32'))
label_tensor = torch.tensor(df['Severity'].values)

dataset = data.TensorDataset(data_tensor, label_tensor)
train_set, test_set = data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10)

# TODO: test on future
# TODO: sort by airport and date

