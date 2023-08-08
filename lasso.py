
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
    mlp_dir = 'mlps'
    mlp_name = 'mlp_1024_layer_11.pt'

    mlp_state_dict = torch.load(os.path.join(mlp_dir, mlp_name), map_location='cpu')

    # out projection matrix
    matrix = mlp_state_dict['fc2.weight'].T
    print(matrix.shape)

