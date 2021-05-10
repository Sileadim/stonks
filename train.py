from datasets import StocksDataset, IGNORE_LIST
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import glob
from model import AutoregressiveLstm





#%%
FILES = list(
    glob.iglob("/home/cehmann/projects/stonks/data/daily/us/nyse stocks/*/*txt")
)

FILES = [ f for f in FILES if f not in IGNORE_LIST]

train_dataset = StocksDataset(files=FILES[:-200])
val_dataset = StocksDataset(files=FILES[-200:-100],sample=False)
train_dataloader  = DataLoader(train_dataset,batch_size=256,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=64,shuffle=False)
model = AutoregressiveLstm().double()
#%%

trainer = pl.Trainer(max_epochs=100,gpus=[0])
trainer.fit(model, train_dataloader, val_dataloader)

# %%
