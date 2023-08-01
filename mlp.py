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
    
    def forward(self, x, return_pre_act=False, return_post_act=False, hidden_noise=0.0):
        h = self.fc1(x)

        if return_pre_act:
            return h
        
        if hidden_noise > 0.0:
            h = h + torch.randn_like(h) * hidden_noise

        h = self.activation(h)

        if return_post_act:
            return h

        h = self.fc2(h)
        return h


def solu(x, temperature=1.0):
    return x * torch.softmax(x * (1/temperature), dim=-1)


class SoluMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x, return_pre_act=False, return_post_act=False, hidden_noise=0.0):
        h = self.fc1(x)

        if return_pre_act:
            return h
        
        if hidden_noise > 0.0:
            h = h + torch.randn_like(h) * hidden_noise

        h = solu(h)
        h = self.ln(h)

        if return_post_act:
            return h

        h = self.fc2(h)
        return h
