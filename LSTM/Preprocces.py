import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torch import optim
import pandas as pd


class WeatherDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.dataset = {}  # {0: (tensor(num_sens=1, len_sen, (word_i, pos_i)=2), tensor(num_sens=1, len_sen))}
        self.convert_sentences_to_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, y = self.dataset[index]
        return X, y

    def convert_sentences_to_dataset(self):
        data_list = [torch.tensor(df_yw.values.astype('float32'), requires_grad=False)
                     for yw, df_yw in self.df.groupby(['year', 'week'])[REL_FEATURES]]
        label_list = [torch.tensor(df_yw.values.astype('float32'), requires_grad=False)
                      for yw, df_yw in self.df.groupby(['year', 'week'])[LABEL]]
        self.dataset = {i: sample_tuple for i, sample_tuple in enumerate(zip(data_list, label_list))}


REL_FEATURES = [' _conds', ' _dewptm', ' _fog', ' _hail', ' _hum', ' _pressurem', ' _rain', ' _snow',
                ' _thunder', ' _tornado', ' _vism', ' _wspdm', 'year', 'month', 'day', 'night', 'morning',
                'noon', 'evening']
LABEL = ['Temp']

df_train = pd.read_csv('../weather_data/df_full_train.csv')
df_test = pd.read_csv('../weather_data/df_full_test.csv')

train_set = data.WeatherDataset(df_train)
test_set = data.WeatherDataset(df_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
