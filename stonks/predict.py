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
        prediction = model.forward(array)
        last_step_prediction = prediction[:,-1].view(1,1)
        if i == 0:
            predictions = last_step_prediction
        else:
            predictions = torch.cat((predictions, last_step_prediction),axis=1)
        input_array = torch.cat((input_array, last_step_prediction),axis=1)

    return predictions

test_dataset = StocksDataset(files=FILTERED[-100:], normalization_func=None)
if args.type == "lstm":
    model = AutoregressiveLstm.load_from_checkpoint(args.path)
else:
    model = Transformer.load_from_checkpoint(args.path)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#%%
for batch in test_dataloader:
    minus_last_ten = batch[:,0:-10]
    std = torch.std(minus_last_ten)
    mean = torch.mean(minus_last_ten)
    normed = (minus_last_ten - mean) / std
    predictions = predict_n_steps(model,normed, n_steps=10)
    real = ((batch - mean) / std).detach().numpy()
    predicted = torch.cat((normed, predictions), axis=1).detach().numpy()
    plt.plot(real.squeeze(0)[-20:],color="blue")
    plt.plot(predicted.squeeze(0)[-20:], color="red")
    plt.show()
    