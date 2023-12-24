import torch
from torch import nn
from models.layers.mlp import MLP
from typing import List


class Critic(nn.Module):

    def __init__(
        self,
        state_encoder_input_dim: int = 3,
        state_encoder_output_dim: int = 64,
        state_encoder_hidden_dims: List[int] = [],
        state_encoder_hidden_act: str = "ReLU",
        state_encoder_out_act: str = "ReLU",
        
        action_encoder_input_dim: int = 1,
        action_encoder_output_dim: int = 64,
        action_encoder_hidden_dims: List[int] = [],
        action_encoder_hidden_act: str = "ReLU",
        action_encoder_out_act: str = "ReLU",
        
        q_estimator_input_dim: int = 128,
        q_estimator_output_dim: int = 1,
        q_estimator_hidden_dims: List[int] = [32],
        q_estimator_hidden_act: str = "ReLU",
        q_estimator_out_act: str = "Identity",
    ):
        super(Critic, self).__init__()
        self.state_encoder = MLP(
            input_dim=state_encoder_input_dim,
            output_dim=state_encoder_output_dim,
            hidden_dims=state_encoder_hidden_dims,
            hidden_act=state_encoder_hidden_act,
            out_act=state_encoder_out_act,
        )
        
        self.action_encoder = MLP(
            input_dim=action_encoder_input_dim,
            output_dim=action_encoder_output_dim,
            hidden_dims=action_encoder_hidden_dims,
            hidden_act=action_encoder_hidden_act,
            out_act=action_encoder_out_act,
        )
        
        self.q_estimator = MLP(
            input_dim=q_estimator_input_dim,
            output_dim=q_estimator_output_dim,
            hidden_dims=q_estimator_hidden_dims,
            hidden_act=q_estimator_hidden_act,
            out_act=q_estimator_out_act,
        )

    def forward(self, x, a):
        emb = torch.cat([self.state_encoder(x), self.action_encoder(a)], dim=-1)
        return self.q_estimator(emb)