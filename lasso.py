
import os
import torch

from sklearn.linear_model import Lasso

from mlp import GeluMLP, SoluMLP


def get_out_proj(matrix, neuron_idx):
    # get out_projection for neuron.
    raise NotImplementedError


def back_to_activations(matrix, out_projection):
    # do lasso.
    raise NotImplementedError


def echo(matrix, neuron_idx):
    out_projection = get_out_proj(matrix, neuron_idx)
    activations = back_to_activations(matrix, out_projection)
    top_activations = activations.topk(10)
    return top_activations


def main():
    mlp_path = "/Users/slava/Desktop/od/8xsolu/2023-08-03_12-44-13/mlp_65536_layer_1.pt"

    mlp_state_dict = torch.load(mlp_path, map_location='cpu')

    matrix = mlp_state_dict['fc2.weight']

    # matrix shape is (d_model, 4 * d_model)
    print(matrix.shape)


if __name__ == "__main__":
    main()
