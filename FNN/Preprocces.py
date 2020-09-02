import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
import pandas as pd

REL_FEATURES = ['Stop', 'Station', 'Visibility(mi)', 'Junction', 'Crossing', 'Wind_Speed(mph)',
       'Pressure(in)', 'Humidity(%)', 'Traffic_Signal', 'Temperature(F)', 'Distance(mi)',
       'Start_Lat', 'Start_Lng', 'Severity', 'Airport_Code', 'City', 'County', 'Street',
       'Start_Time', 'End_Time', 'Civil_Twilight']

df_train = pd.read_csv('train.csv')
df_train = df_train[REL_FEATURES + ['Severity']]
df_test = pd.read_csv('test.csv')
df_test = df_test[REL_FEATURES + ['Severity']]

data_tensor_train = torch.tensor(df_train.drop('Severity', axis=1).values.astype('float32'))
label_tensor_train = torch.tensor(df_train['Severity'].values)
data_tensor_test = torch.tensor(df_test.drop('Severity', axis=1).values.astype('float32'))
label_tensor_test = torch.tensor(df_test['Severity'].values)

train_set = data.TensorDataset(data_tensor_train, label_tensor_train)
test_set = data.TensorDataset(data_tensor_test, label_tensor_test)
#train_set, test_set = data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])


train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10)
# TODO: test on future
# TODO: sort by airport and date

