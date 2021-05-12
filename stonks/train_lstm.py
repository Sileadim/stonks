from pytorch_lightning.utilities.cli import LightningCLI
from model import AutoregressiveLstm, Transformer

from datasets import StocksDataModule


cli = LightningCLI(AutoregressiveLstm, StocksDataModule)
