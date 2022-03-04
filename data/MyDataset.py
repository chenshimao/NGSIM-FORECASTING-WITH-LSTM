from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from utils.tools import StandardScaler


class Dataset_NGSIM(Dataset):
    def __init__(self, flag, seq_len, feature_x, feature_y):
        self.flag = flag
        self.seq_len = seq_len
        self.feature_x = feature_x
        self.feature_y = feature_y
        self.feature = list(set(feature_x + feature_y))
        self.scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw =pd.read_csv(r"data/data.csv")
        df_data = df_raw[self.feature]
        self.scaler.fit(df_data)
        data = self.scaler.transform(df_data)
        num_train = int(len(data) * 0.8)
        num_test = len(data) - num_train
        if self.flag == 'train':
            self.data_x = data[self.feature_x].values[:num_train]
            self.data_y = data[self.feature_y].values[:num_train]
        if self.flag == 'test':
            self.data_x = data[self.feature_x].values[-num_test:]
            self.data_y = data[self.feature_y].values[-num_test:]

    def __getitem__(self, index):
        s_begin = index
        s_end = index + self.seq_len
        return torch.FloatTensor(self.data_x[s_begin:s_end]), torch.FloatTensor(self.data_y[s_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len


