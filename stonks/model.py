#%%
from datasets import StocksDataset, IGNORE_LIST
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import glob
from torch import Tensor, nn
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau


class AutoregressiveBase(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        # batchsize, n_features, length
        x = batch[:, :, 0:-1]
        y = batch[:, :, 1:]
        y_pred = self.forward(x)
        loss = self.loss_func(y, y_pred)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:, :, 0:-1]
        y = batch[:, :, 1:]
        y_pred = self.forward(x)
        loss = self.loss_func(y, y_pred)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
        #return {
        #    "optimizer": optimizer,
        #    "lr_scheduler": {
        #        "scheduler": ReduceLROnPlateau(optimizer, patience=50),
        #        "monitor": "train_loss",
        #    },
        # }


class AutoregressiveLstm(AutoregressiveBase):
    def __init__(
        self,
        hidden_size=256,
        n_features=1,
        use_convolutions=False,
        conv_out_features=None,
        num_layers=2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.use_convolutions = use_convolutions
        self.conv_out_features = (
            conv_out_features if conv_out_features else self.n_features
        )
        self.lstm_input_size = n_features
        self.num_layers = num_layers
        if self.use_convolutions:
            self.convolutions = ConvolutionalFilters(
                n_features=self.n_features, out_features=self.conv_out_features
            ).double()
            self.lstm_input_size += self.conv_out_features
        self.encoder = torch.nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        ).double()
        self.linear = torch.nn.Linear(self.hidden_size, self.n_features).double()
        self.loss_func = torch.nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):

        batch_size, n_features, length = x.shape
        if self.use_convolutions:
            lstm_input = self.convolutions(x)
        else:
            lstm_input = x
        encoded, _ = self.encoder(lstm_input.transpose(2, 1))
        out = self.linear(encoded).view(batch_size, n_features, length)
        return out


class PositionalEncoding(pl.LightningModule):
    def __init__(self, hidden_size, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        pe = torch.zeros(max_len, self.hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2).float()
            * (-math.log(10000.0) / self.hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose()
        self.register_buffer("pe", pe)
        self.save_hyperparameters()

    def forward(self, x):

        batch_size, n_features, length = x.shape
        positional_encodings = (
            self.pe[:length, : self.hidden_size - n_features]
            .view(1, length, -1)
            .repeat(batch_size, 1, 1)
        )
        x = torch.cat((x.transpose(2, 1), positional_encodings), axis=2)
        # x = torch.cat((x.view(batch_size, length, -1), torch.zeros(length).view(1,length,1).repeat(batch_size,1,self.hidden_size-1).to(self.device)), axis=2)

        return x


class Transformer(AutoregressiveBase):
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

    def __init__(
        self,
        hidden_size=256,
        num_layers=1,
        nhead=8,
        dim_feedforward=256,
        n_features=5,
        use_convolutions=False,
        conv_out_features=None,

    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.n_features = n_features
        self.dim_feedforward = dim_feedforward
        self.use_convolutions = use_convolutions
        self.conv_out_features = (
            conv_out_features if conv_out_features else self.n_features
        )

        if self.use_convolutions:
            self.convolutions = ConvolutionalFilters(
                n_features=self.n_features, out_features=self.conv_out_features
            ).double()

        self.positional_encoding = PositionalEncoding(
            hidden_size=self.hidden_size
        ).double()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers, norm=self.layer_norm
        ).double()
        self.linear = torch.nn.Linear(self.hidden_size, self.n_features).double()
        self.loss_func = torch.nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):

        # seq_length,batch_size,
        batch_size, n_features, length = x.shape

        if self.use_convolutions:
            pos_input = self.convolutions(x)
        else:
            pos_input = x

        positionally_encoded = self.positional_encoding(pos_input)
        mask = self.generate_square_subsequent_mask(length).to(self.device)
        encoded = self.transformer_encoder(
            positionally_encoded.view(length, batch_size, self.hidden_size), mask=mask
        )
        out = self.linear(encoded)
        return out.view(batch_size, n_features, length)


class ConvolutionalFilters(pl.LightningModule):
    def __init__(
        self,
        n_features=5,
        out_features=5,
        small_kernel_size=5,
        big_kernel_size=20,
        pad_type="same",
    ):
        super().__init__()
        self.n_features = n_features
        self.out_features = out_features
        self.small_kernel_size = small_kernel_size
        self.big_kernel_size = big_kernel_size
        self.pad_type = pad_type
        self.small_filter = torch.nn.Conv1d(
            self.n_features,
            self.out_features,
            self.small_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.big_filter = torch.nn.Conv1d(
            self.n_features,
            self.out_features,
            self.big_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.save_hyperparameters()

    def forward(self, x):

        first_value = x[:, :, 0:1]
        padded_for_small = torch.cat(
            (first_value.repeat(1, 1, self.small_kernel_size - 1), x), axis=-1
        )
        padded_for_big = torch.cat(
            (first_value.repeat(1, 1, self.big_kernel_size - 1), x), axis=-1
        )

        small_filtered = self.small_filter(padded_for_small)
        big_filtered = self.big_filter(padded_for_big)
        summed = small_filtered + big_filtered
        return torch.cat((x, summed), axis=1)
