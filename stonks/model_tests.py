#%%
from datasets import StocksDataset, IGNORE_LIST
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import glob
from model import AutoregressiveLstm


FILES = list(
    glob.iglob("/home/cehmann/projects/stonks/data/daily/us/nyse stocks/*/*txt")
)

FILES = [ f for f in FILES if f not in IGNORE_LIST]

train_dataset = StocksDataset(files=FILES[:10])
train_dataloader = DataLoader(train_dataset,batch_size=2,shuffle=False)

model = AutoregressiveLstm().double()
for batch in train_dataloader:

    model.training_step(batch.double(),0)
