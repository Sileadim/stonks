#%%
from jsonargparse.util import change_to_path_dir
from torch.nn.modules import normalization
from datasets import CryptoDataset, FILTERED, DataLoader
from model import AutoregressiveLstm, Transformer
from jsonargparse import ArgumentParser
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import os
import matplotlib
import pandas as pd
matplotlib.use("tkagg")


def predict_n_steps(model, array, n_steps=5):

    input_array = array
    predictions = None
    for i in range(n_steps):
        prediction = model.forward(input_array)
        last_step_prediction = prediction[:, :, -1].unsqueeze(-1)
        input_array = torch.cat((input_array, last_step_prediction), axis=2)

    return input_array
    
    


def load_params(p):
    params_path = os.path.join(os.path.dirname(os.path.dirname(p)), "config.yaml")
    params_dict = yaml.load(open(params_path))
    del params_dict["data"]["files"]
    return params_dict


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--type", type=str)
    parser.add_argument("--prediction_type",type=str, default="predict_n_steps")
    parser.add_argument("--predict_n_steps", type=int, default=10)
    parser.add_argument("--show_last_n_steps", type=int, default=30)
    args = parser.parse_args()
    params_dict = load_params(args.path)
    test_dataset = CryptoDataset(
        [pd.read_csv("/home/cehmann/workspaces/stonks/data/5 min/world/cryptocurrencies/bts.v.txt").iloc[-1000:].reset_index()],
        sample_length=params_dict["data"]["sample_length"],
    )

    if args.type == "lstm":
        model = AutoregressiveLstm.load_from_checkpoint(args.path)
    else:
        model = Transformer.load_from_checkpoint(args.path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for batch in test_dataloader:

        if args.prediction_type=="predict_n_steps":

            minus_last_ten = batch[:, :, 0:-args.predict_n_steps]
            normed = batch
            predictions = (
                predict_n_steps(model, minus_last_ten, n_steps=args.predict_n_steps)
                .detach()
                .numpy()
            )

            for i, name in enumerate(["<CLOSE>"]):
                plt.plot(normed[0, i, -args.show_last_n_steps :], color="blue")
                plt.plot(
                    predictions[0, i, -args.show_last_n_steps :],
                    color="red",
                )
                plt.title(name)
                plt.show()

        else:
            predictions = model.forward(batch).detach().numpy()
            print(batch.shape)
            print(predictions.shape)
            plt.plot(batch[0,0,1:][-100:])
            plt.plot(predictions[0,0,:-1][-100:])
            plt.show()

