#%%
from datasets import StocksDataset, IGNORE_LIST
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import glob
from model import AutoregressiveLstm,Transformer,ConvolutionalFilters
import unittest


FILES = list(
    glob.iglob("data/daily/us/nyse stocks/*/*txt")
)
FILES = [ f for f in FILES if f not in IGNORE_LIST]

class Test(unittest.TestCase):

    def test_vanilla_transformer(self):

        train_dataset = StocksDataset(files=FILES[:10],min_length=30)
        train_dataloader = DataLoader(train_dataset,batch_size=2,shuffle=False)

        model = Transformer().double()
        for batch in train_dataloader:
            model.training_step(batch.double(), 0)
            break

    def test_vanilla_lstm(self):

        train_dataset = StocksDataset(files=FILES[:10],min_length=30)
        train_dataloader = DataLoader(train_dataset,batch_size=2,shuffle=False)

        model = AutoregressiveLstm().double()
        for batch in train_dataloader:
            model.training_step(batch.double(), 0)
            break

    def test_convolution(self):
        model = ConvolutionalFilters().double()
        sample = torch.arange(start=0,end=60).view(1,1,-1).repeat(2,5,1).double()
        out = model.forward(sample)
        self.assertEqual(out.shape, torch.Size([2, 10, 60]))

    def test_lstm_with_convolution(self):
        train_dataset = StocksDataset(files=FILES[:10],min_length=30)
        train_dataloader = DataLoader(train_dataset,batch_size=2,shuffle=False)

        model = AutoregressiveLstm(use_convolutions=True).double()
        for batch in train_dataloader:
            model.training_step(batch.double(), 0)
            break

if __name__ == '__main__':
    unittest.main()
