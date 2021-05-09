import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

files = list(
    glob.iglob("/home/cehmann/projects/stonks/data/daily/us/nyse stocks/*/*txt")
)


def subtract_mean_and_divide_by_std(array):
    return (array - np.mean(array)) / np.std(array)


class StocksDataset(Dataset):
    def __init__(
        self,
        files,
        min_length=365,
        normalization_func=subtract_mean_and_divide_by_std,
        column="<CLOSE>",
    ):
        self.data = []
        self.min_length = min_length
        self.normalization_func = normalization_func
        self.column = column
        for p in files:
            try:
                d = pd.read_csv(p)
                if len(d) >= min_length:
                    self.data.append(d.iloc[-min_length:])
            except Exception as e:
                print(e)
                pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        array = np.array(self.data[idx][self.column])
        if self.normalization_func:
            array = self.normalization_func(array)
        return array


class RandomWalkDataset(Dataset):
    def __init__(
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
        self.data = [
            self.random_walk_with_drift(
                n_steps=self.n_steps,
                sigma=self.sigma,
                drift=self.drift,
                start=self.start,
            )
        ]

    def __len__(self):
        return self.length

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


dataset = StocksDataset(files=files[0:10])
dataset = RandomWalkDataset()

print(len(dataset[0]))

