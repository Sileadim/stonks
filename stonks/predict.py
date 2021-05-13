#%%
from jsonargparse.util import change_to_path_dir
from torch.nn.modules import normalization
from datasets import StocksDataset, FILTERED, DataLoader
from model import AutoregressiveLstm, Transformer
from jsonargparse import ArgumentParser
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import os
import matplotlib

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

    args = parser.parse_args()
    params_dict = load_params(args.path)
    test_dataset = StocksDataset(
        files=FILTERED[-100:],
        min_length=180,
        columns=params_dict["data"]["columns"],
        normalization=params_dict["data"]["normalization"],
    )

    if args.type == "lstm":
        model = AutoregressiveLstm.load_from_checkpoint(args.path)
    else:
        model = Transformer.load_from_checkpoint(args.path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for batch in test_dataloader:
        minus_last_ten = batch[:, :, 0:-10]
        normed = batch
        predictions = (
            predict_n_steps(model, minus_last_ten, n_steps=10).detach().numpy()
        )

        for i, name in enumerate(params_dict["data"]["columns"]):
            plt.plot(normed[0, i, :], color="blue")
            plt.plot(
                predictions[
                    0,
                    i,
                ],
                color="red",
            )
            plt.title(name)
            plt.show()
