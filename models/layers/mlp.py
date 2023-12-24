import torch
import torch.nn as nn
from typing import List
from models.layers.utils import get_activation


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 32],
        hidden_act: str = "ReLU",
        out_act: str = "Identity",
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.hidden_act = get_activation(hidden_act)
        self.out_act = get_activation(out_act)

        input_dims = [input_dim] + hidden_dims
        output_dims = hidden_dims + [output_dim]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            self.layers.append(nn.Linear(in_dim, out_dim))
            
            is_last = (i == len(input_dims) - 1)
            self.layers.append(self.out_act if is_last else self.hidden_act)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            xs = layer(xs)
        return xs