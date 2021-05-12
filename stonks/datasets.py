#%%
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import warnings
import warnings


IGNORE_LIST = [
    "/home/cehmann/workspaces/stonks/data/daily/us/nyse stocks/2/zexit.us.txt"
]
#%%
FILES = list(
    glob.iglob("/home/cehmann/workspaces/stonks/data/daily/us/nyse stocks/*/*txt")
)

FILTERED = [f for f in FILES if f not in IGNORE_LIST]
#%%
def subtract_mean_and_divide_by_std(array):
    zero_mean = array - np.mean(array)
    std = np.std(array)
    if std:
        return zero_mean / np.std(array)
    return zero_mean


class StocksDataset(Dataset):
    def __init__(
        self,
        files=FILES[:4],
        min_length=365,
        normalization_func=subtract_mean_and_divide_by_std,
        columns=["<VOL>","<OPEN>","<HIGH>","<LOW>","<CLOSE>"],
        sample=True,
    ):
        self.data = []
        self.min_length = min_length
        self.normalization_func = normalization_func
        self.columns = columns
        self.sample = sample
        for p in files:
            try:
                d = pd.read_csv(p)
            except Exception as e:
                print(p, e)
                continue
            if len(d) >= min_length:
                data = [ np.array(d[c]) for c in self.columns ]
                self.data.append((p, data))

    def sample_from(self, data):

        start = np.random.randint(low=0, high=len(data[0]) - self.min_length + 1)
        return [  d[start : start + self.min_length] for d in data]

    def process_data(self, data):

        if self.sample:
            sample = self.sample_from(data)
        else:
            sample = [  d[-self.min_length :] for d in data]
        if self.normalization_func:
            stacked = np.stack(  [self.normalization_func(array) for array in sample])
        return stacked

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, sample = self.data[idx]
        try:
            processed = self.process_data(sample)
        except RuntimeWarning as w:
            print("Failing ", path)
            raise w
        return processed


class RandomWalkDataset(Dataset):
    def __init__(
        self,
        length=100,
        n_steps=365,
        sigma=1,
        drift=0.01,
        start=100,
        normalization_func=subtract_mean_and_divide_by_std,
    ):

        self.length = length
        self.n_steps = n_steps
        self.sigma = sigma
        self.drift = drift
        self.start = start
        self.normalization_func = normalization_func
        self.data = [
            self.random_walk_with_drift(
                n_steps=self.n_steps,
                sigma=self.sigma,
                drift=self.drift,
                start=self.start,
            )
            for i in range(self.length)
        ]

    def __len__(self):
        return self.length

    @staticmethod
    def random_walk_with_drift(n_steps=365, sigma=1, drift=0.01, start=0):

        values = []
        x = start
        values.append(x)
        for i in range(n_steps - 1):
            x = drift + x + np.random.normal(scale=sigma)
            values.append(x)
        a = np.array(values)
        return a

    def __getitem__(self, idx):

        array = self.data[idx]
        if self.normalization_func:
            array = self.normalization_func(array)
        return array

    

class StocksDataModule(pl.LightningDataModule):
    def __init__(self, files=FILTERED, train_batch_size=64, val_batch_size=64,min_length=365):
        super().__init__()
        self.files = files
        self.train_batch_size = 64
        self.val_batch_size = 64
        self.min_length = min_length

    def setup(self, stage):
        pass

    def prepare_data(self):
        self.train_split = StocksDataset(files=self.files[:-200],min_length=self.min_length)
        self.val_split = StocksDataset(files=self.files[-200:-100],min_length=self.min_length)

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.val_batch_size)

    def teardown(self):
        pass

