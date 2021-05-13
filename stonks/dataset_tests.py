#%%
from datasets import StocksDataset, IGNORE_LIST
import torch
from torch.utils.data import DataLoader
import glob
import unittest


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


if __name__ == '__main__':
    unittest.main()
