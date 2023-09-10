import torch
import numpy as np
from torch import nn
from utils import ReplayMemory

class DoubleDQN(nn.Module):
    def __init__(self, env, config):
        super(DoubleDQN, self).__init__()
        self.config = config
        self.replay_memory = ReplayMemory(self.config)

        d_state = env.observation_space.shape[0]
        n_action = env.action_space.n

        self.network = nn.Sequential(
            nn.Linear(d_state, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, n_action)
        )

        self.target_network = nn.Sequential(
            nn.Linear(d_state, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, n_action)
        )

        for param in self.target_network.parameters():
            param.requires_grad = False

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            params=self.network.parameters(),
            lr=self.config.lr, 
            weight_decay=1e-3
        )

    def forward(self, x):
        Qs = self.network(x)
        return Qs

    def forward_target_network(self, x):
        Qs = self.target_network(x)
        return Qs

    def get_argmax_action(self, x):
        s = torch.from_numpy(x).reshape(1, -1).float()
        Qs = self.forward(s)
        argmax_action = Qs.argmax(dim=-1).item()
        return argmax_action

    def train(self):
        transitions = self.replay_memory.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_array = np.stack(states, axis=0)  # (n_batch, d_state)
        actions_array = np.stack(actions, axis=0, dtype=np.int64)  # (n_batch)
        rewards_array = np.stack(rewards, axis=0)  # (n_batch)
        next_states_array = np.stack(next_states, axis=0)  # (n_batch, d_state)
        dones_array = np.stack(dones, axis=0)  # (n_batch)

        states_tensor = torch.from_numpy(states_array).float()  # (n_batch, d_state)
        actions_tensor = torch.from_numpy(actions_array)  # (n_batch)
        rewards_tensor = torch.from_numpy(rewards_array).float()  # (n_batch)
        next_states_tensor = torch.from_numpy(next_states_array).float()  # (n_batch, d_state)
        dones_tensor = torch.from_numpy(dones_array).float()  # (n_batch)

        Qs = self.forward(states_tensor)  # (n_batch, n_action)
        with torch.no_grad():
            next_Qs = self.forward(next_states_tensor)  # (n_batch, n_action)
        next_target_Qs = self.forward_target_network(next_states_tensor)  # (n_batch, n_action)

        chosen_Q = Qs.gather(dim=-1, index=actions_tensor.reshape(-1, 1)).reshape(-1)  # (n_batch, 1) -> (n_batch)
        next_argmax_actions = next_Qs.argmax(dim=-1).reshape(-1, 1)
        next_target_max_Q = next_target_Qs.gather(dim=-1, index=next_argmax_actions).reshape(-1)
        target_Q = rewards_tensor + (1 - dones_tensor) * self.config.gamma * next_target_max_Q
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(chosen_Q, target_Q)

        # Update by gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()