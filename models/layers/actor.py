from torch import nn
from models.layers.mlp import MLP
from typing import List


class Actor(nn.Module):

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_dims: List[int] = [128, 64],
        hidden_act: str = "ReLU",
        out_act: str = "Identity",
        action_scaler: float = 2.0, # for Pendulum v0
    ):
        super(Actor, self).__init__()
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            hidden_act=hidden_act,
            out_act=out_act,
        )
        self.action_scaler = action_scaler

    def forward(self, state):
        # Action space of Pendulum v0 is [-2.0, 2.0]
        return self.mlp(state).clamp(-self.action_scaler, self.action_scaler)
