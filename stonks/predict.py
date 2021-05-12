#%%
from datasets import StocksDataset, FILTERED, DataLoader
from model import AutoregressiveLstm, Transformer
from jsonargparse import ArgumentParser
import numpy as np
import torch
import matplotlib.pyplot as plt
parser = ArgumentParser()
parser.add_argument("--path",type=str)
parser.add_argument("--type",type=str)

args = parser.parse_args()
def predict_n_steps(model, array, n_steps=5):

    input_array = array
    predictions = None
    for i in range(n_steps):
        prediction = model.forward(input_array)
        last_step_prediction = prediction[:,:,-1].unsqueeze(-1)
        input_array = torch.cat((input_array, last_step_prediction),axis=2)

    return input_array

test_dataset = StocksDataset(files=FILTERED[-100:],min_length=180)
if args.type == "lstm":
    model = AutoregressiveLstm.load_from_checkpoint(args.path)
else:
    model = Transformer.load_from_checkpoint(args.path)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#%%
for batch in test_dataloader:
    minus_last_ten = batch[:,:,0:-10]
    #std = torch.std(minus_last_ten,axis=2)
    #mean = torch.mean(minus_last_ten,axis=2)
    normed = batch
    #normed = (minus_last_ten - mean) / std
    predictions = predict_n_steps(model,minus_last_ten, n_steps=10).detach().numpy()
    #real = ((batch - mean) / std).detach().numpy()
    for i, name in enumerate(["<VOL>","<OPEN>","<HIGH>","<LOW>","<CLOSE>"]):
        plt.plot(normed[0,i,:],color="blue")
        plt.plot(predictions[0,i,], color="red")
        plt.title(name)
        plt.show()
    
