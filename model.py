#%%
from datasets import StocksDataset
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import glob
class AutoregressiveLstm(pl.LightningModule):

    def __init__(self,hidden_size=256,batch_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.encoder = torch.nn.LSTM(input_size=1,hidden_size=self.hidden_size,num_layers=2,dropout=0.2)
        self.linear = torch.nn.Linear(self.hidden_size, 1)
        self.loss_func = torch.nn.MSELoss()
    def forward(self,x):


        #seq_length,batch_size,
        batch_size, length = x.shape
        encoded,_ = self.encoder(x.view(length,batch_size,1).double())
        out = self.linear(encoded[-1,:,:].view(batch_size,self.hidden_size))
        return out
    

    def training_step(self,batch,batch_idx):

        x = batch[:,0:-1]
        y = batch[:,-1:]
        y_pred = self.forward(batch)
        loss = self.loss_func(y, y_pred)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x = batch[:,0:-1]
        y = batch[:,-1:]
        y_pred = self.forward(batch)
        loss = self.loss_func(y, y_pred)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)



#%%
FILES = list(
    glob.iglob("/home/cehmann/projects/stonks/data/daily/us/nyse stocks/*/*txt")
)


train_dataset = StocksDataset(files=FILES[:-200])
val_dataset = StocksDataset(files=FILES[-200:-100],sample=False)
train_dataloader  = DataLoader(train_dataset,batch_size=4,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=4,shuffle=False)
model = AutoregressiveLstm().double()
#%%

trainer = pl.Trainer(max_epochs=100,gpus=[0])
trainer.fit(model, train_dataloader, val_dataloader)

# %%
