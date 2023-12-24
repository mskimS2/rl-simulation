import torch
import numpy as np
from torch import nn


class OrnsteinUhlenbeckProcess:
    """
    OU process; The original implementation is provided by minimalRL.
    https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    """

    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = (
            self.x_prev + 
            self.theta * (self.mu - self.x_prev) * self.dt + 
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x


class DDPG(nn.Module):
    def __init__(
        self,
        critic: nn.Module,
        critic_target: nn.Module,
        actor: nn.Module,
        actor_target: nn.Module,
        lr_critic: float = 0.0005,
        lr_actor: float = 0.001,
        gamma: float = 0.99,
    ):
        """
        parameters
        - critic: critic network
        - critic_target: critic network target
        - actor: actor network
        - actor_target: actor network target
        - lr_critic: learning rate of critic
        - lr_actor: learning rate of actor
        - gamma: discount factor
        """

        super(DDPG, self).__init__()
        self.critic = critic
        self.actor = actor
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.gamma = gamma

        self.critic_opt = torch.optim.Adam(
            params=self.critic.parameters(),
            lr=lr_critic,
            betas=(0.9, 0.999),
        )
        self.actor_opt = torch.optim.Adam(
            params=self.actor.parameters(),
            lr=lr_actor,
            betas=(0.9, 0.999),
        )

        # setup target networks
        critic_target.load_state_dict(critic.state_dict())
        self.critic_target = critic_target
        actor_target.load_state_dict(actor.state_dict())
        self.actor_target = actor_target

        self.criteria = nn.SmoothL1Loss()

    @torch.no_grad()
    def get_action(self, state):
        return self.actor(state)

    def update(
        self, 
        s: torch.Tensor, 
        a: torch.Tensor, 
        r: torch.Tensor, 
        ns: torch.Tensor, 
        done: torch.Tensor, 
    ):
        """
        parameters
        - s: state
        - a: action
        - r: reward
        - ns: next state
        """

        # compute critic loss and update the critic parameters
        with torch.no_grad():
            critic_target = r + self.gamma * self.critic_target(ns, self.actor_target(ns)) * (1 - done)
        critic_loss = self.criteria(self.critic(s, a), critic_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # compute actor loss and update the actor parameters
        actor_loss = -self.critic(s, self.actor(s)).mean()  # !!!! Impressively simple
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()