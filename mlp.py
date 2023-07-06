import torch
import torch.nn as nn


class GeluMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()
    
    def forward(self, x, return_pre_act=False, return_post_act=False):
        h = self.fc1(x)

        if return_pre_act:
            return h

        h = self.activation(h)

        if return_post_act:
            return h

        h = self.fc2(h)
        return h
