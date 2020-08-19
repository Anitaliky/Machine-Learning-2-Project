import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
import pandas as pd


df = pd.read_csv('US_Accidents_June20.csv')

flot = ['Start_Lat', 'Start_Lng', 'Distance(mi)', 'Temperature(F)', 'Humidity(%)',
        'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
bol = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
       'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
       'Traffic_Signal', 'Turning_Loop']
df = df[flot+bol+['Severity']].dropna()
print(df.info())
print(len(df))
df = df.head(1000000)
df.loc[df['Severity'].isin([1, 2]), 'Severity'] = 0
df.loc[df['Severity'].isin([3, 4]), 'Severity'] = 1

data_tensor = torch.tensor(df.drop('Severity', axis=1).values.astype('float32'))
label_tensor = torch.tensor(df['Severity'].values)

dataset = data.TensorDataset(data_tensor, label_tensor)
train_set, test_set = data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10)

# TODO: test on future
# TODO: sort by airport and date

