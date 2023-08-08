
import os
import torch

from sklearn.linear_model import Lasso

from mlp import GeluMLP, SoluMLP


def get_out_proj(matrix, neuron_idx):
    # get out_projection for neuron.
    raise NotImplementedError


def back_to_activations(matrix, out_projection):
    # Perform Lasso regression using the given out_projection.
    lasso = Lasso(alpha=0.1, max_iter=1000)
    lasso.fit(matrix, out_projection)
    activations = torch.tensor(lasso.coef_)
    return activations


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

    neuron_idx = 5  # Example index
    top_activations = echo(matrix, neuron_idx)
    print(top_activations)  # Print the top 10 activations


if __name__ == "__main__":
    main()
