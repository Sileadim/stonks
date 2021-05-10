#%%
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error")
#%%
FILES = list(
    glob.iglob("/home/cehmann/projects/stonks/data/daily/us/nyse stocks/*/*txt")
)

#%%
def subtract_mean_and_divide_by_std(array):
    return (array - np.mean(array)) / np.std(array)


class StocksDataset(Dataset):
    def __init__(
        self,
        files=FILES[:4],
        min_length=365,
        normalization_func=subtract_mean_and_divide_by_std,
        column="<CLOSE>",
        sample = True
    ):
        self.data = []
        self.min_length = min_length
        self.normalization_func = normalization_func
        self.column = column
        self.sample = sample
        for p in files:
            try:
                d = pd.read_csv(p)
               
                if len(d) >= min_length:
                    print("length",len(d))
                    self.data.append((p,d[self.column]))
            except Exception as e:
                print(e)
                pass
    
    def sample_from(self,d):

        start = np.random.randint(low=0,high=len(d)-self.min_length+1)
        return d[start:start+self.min_length]

    def process_data(self,d):

        if self.sample:
            sample = self.sample_from(d)
        else:
            sample = d[-self.min_length:]
        array = np.array(sample)
        if self.normalization_func:
            try:

                array = self.normalization_func(array)
            except RuntimeWarning as w:
                print("Failing")
                print(idx)

        return array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path,sample = self.data[idx]
        print(path)
        return self.process_data(sample)


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


#%%
print(len(FILES))
#%%
dataset = StocksDataset(files=FILES)
#%%
for idx, batch in enumerate(dataset):
    print(idx)
    pass
#%%
##dataset = RandomWalkDataset()
#
#plt.plot(dataset[0])
#plt.show()
#
#
#
#data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
#
for idx, batch in enumerate(data_loader):
    #print(idx,batch.shape)
    break
## %%

# %%
