from pytorch_lightning.utilities.cli import LightningCLI
from model import AutoregressiveLstm, Transformer
import pytorch_lightning as pl
from datasets import StocksDataModule



from pytorch_lightning.callbacks import LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval='epoch')
#trainer = Trainer(callbacks=[lr_monitor])

cli = LightningCLI(Transformer, StocksDataModule)
