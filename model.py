#%%
from datasets import StocksDataset, IGNORE_LIST
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import glob
from torch import Tensor


class AutoregressiveLstmSimple(pl.LightningModule):
    def __init__(self, hidden_size=256, batch_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.encoder = torch.nn.LSTM(
            input_size=1, hidden_size=self.hidden_size, num_layers=2, dropout=0.2
        )
        self.linear = torch.nn.Linear(self.hidden_size, 1)
        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):

        # seq_length,batch_size,
        batch_size, length = x.shape
        encoded, _ = self.encoder(x.view(length, batch_size, 1).double())
        out = self.linear(encoded[-1, :, :].view(batch_size, self.hidden_size))
        return out

    def training_step(self, batch, batch_idx):

        x = batch[:, 0:-1]
        y = batch[:, -1:]
        y_pred = self.forward(batch)
        loss = self.loss_func(y, y_pred)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:, 0:-1]
        y = batch[:, -1:]
        y_pred = self.forward(batch)
        loss = self.loss_func(y, y_pred)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class AutoregressiveLstm(pl.LightningModule):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = torch.nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        ).double()
        self.linear = torch.nn.Linear(self.hidden_size, 1).double()
        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):

        # seq_length,batch_size,
        batch_size, length = x.shape
        encoded, _ = self.encoder(x.view(batch_size, length, 1))
        out = self.linear(encoded).view(batch_size, length)
        return out

    def training_step(self, batch, batch_idx):

        x = batch[:, 0:-1]
        y = batch[:, 1:]
        y_pred = self.forward(x)
        loss = self.loss_func(y, y_pred)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:, 0:-1]
        y = batch[:, 1:]
        y_pred = self.forward(x)
        loss = self.loss_func(y, y_pred)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(pl.LightningModule):
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def __init__(self, hidden_size=256, batch_size=4, num_layers=4):

        self.hidden_size = hidden_size
        self.batch_size = 4
        self.num_layers = num_layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=8
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.positional_encoding = PositionalEncoding(d_model=self.hidden)

    def forward(self, x):

        # seq_length,batch_size,
        batch_size, length = x.shape
        encoded, _ = self.encoder(x.view(length, batch_size, 1).double())
        out = self.linear(encoded[-1, :, :].view(batch_size, self.hidden_size))
        return out
