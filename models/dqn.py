import torch
import numpy as np
from torch import nn


class DQN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        qnet: nn.Module,
        qnet_target: nn.Module,
        lr: float,
        gamma: float,
        epsilon: float,
    ):
        """
        parameters
        - state_dim: input state dimension
        - action_dim: action dimension
        - qnet: main q network
        - qnet_target: target q network
        - lr: learning rate
        - gamma: discount factor of MDP
        - epsilon: E-greedy factor
        """
        super(DQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer("epsilon", torch.ones(1) * epsilon)

        # target network related
        qnet_target.load_state_dict(qnet.state_dict())
        self.qnet_target = qnet_target
        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)
        if torch.from_numpy(prob).float() <= self.epsilon:  # random policy
            return int(np.random.choice(range(self.action_dim)))
            
        qs = self.qnet(state)
        return int(qs.argmax(dim=-1)) # greedy policy

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute Q-Learning target with `target network`
        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims=True)
            q_target = r + self.gamma * q_max * (1 - done)

        q_val = self.qnet(s).gather(1, a)
        loss = self.criteria(q_val, q_target)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()