import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torch import optim
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


class WeatherDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.dataset = []
        self.convert_sentences_to_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, y = self.dataset[index]
        return X, y

    def convert_sentences_to_dataset(self):
        for yw, df_yw in self.df.groupby(['sequence']):
            data_ten = torch.tensor(df_yw[REL_FEATURES].values.astype('float32'), requires_grad=False)
            label_ten = torch.tensor(df_yw[LABEL].values.astype('float32'), requires_grad=False)
            self.dataset.append((data_ten, label_ten))


# ' _conds', embbeding?
# 'year', 'month', 'day'
REL_FEATURES = [' _dewptm', ' _fog', ' _hail', ' _hum', ' _pressurem', ' _rain', ' _snow',
                ' _thunder', ' _tornado', ' _vism', ' _wspdm', 'night', 'morning', 'year', 'month', 'day',
                'noon', 'evening', 'month_cos', 'month_sin', 'hour_cos', 'hour_sin', 'week_cos', 'week_sin']
LABEL = ['Temp']

df_train = pd.read_csv('../weather_data/Sequence_data_frames/df_full_all_train.csv')
df_test = pd.read_csv('../weather_data/Sequence_data_frames/df_full_all_test.csv')
print()
# df_train = df_train[df_train['month'] == 5]

train_set = WeatherDataset(df_train)
test_set = WeatherDataset(df_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
