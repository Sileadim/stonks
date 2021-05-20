#%%
from datasets import StocksDataset, IGNORE_LIST, CryptoDataset
import torch
from torch.utils.data import DataLoader
import glob
import unittest
import pandas as pd

FILES = sorted(list(
    glob.iglob("data/daily/us/nyse stocks/*/*txt")
))
FILES = [ f for f in FILES if f not in IGNORE_LIST]

class Test(unittest.TestCase):

    def test_single_column(self):
        dataset = StocksDataset(files=FILES[:2],min_length=3,columns=["<CLOSE>"],sample=False)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        for batch in dataloader:
            self.assertEqual(batch.shape, torch.Size([1,1,3]))
            break

    def test_all_columns(self):
        dataset = StocksDataset(files=FILES[:2],min_length=3,sample=False)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        for batch in dataloader:
            self.assertEqual(batch.shape, torch.Size([1,5,3]))
            break

class CryptoTest(unittest.TestCase):

    def test_crypto(self):

        path = "data/5 min/world/cryptocurrencies/bts.v.txt"
        df = pd.read_csv(path)
        dataset = CryptoDataset(dfs=[df.iloc[:-2000].reset_index()])
        print(dataset[0])



if __name__ == '__main__':
    unittest.main()
