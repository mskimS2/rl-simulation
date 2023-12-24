import gym
import torch

from config import ddpg_pendulum_config as args
from utils.train_utils import to_tensor
from models.layers.actor import Actor
from models.layers.critic import Critic
from models.ddpg import DDPG


if __name__ == "__main__":
    actor, actor_target = Actor(), Actor()
    critic, critic_target = Critic(), Critic()
    agent = DDPG(
        critic=critic,
        critic_target=critic_target,
        actor=actor,
        actor_target=actor_target,
    )
    agent.load_state_dict(torch.load("results/models/ddpg_pendulum_v1.pth"))
    agent.to(args.device)
        
    env = gym.make("Pendulum-v1")
    s = env.reset()
    cum_r = 0

    while True:
        s = to_tensor(s, size=(1, 3)).to(args.device)
        a = agent.get_action(s).to("cpu").numpy()
        ns, r, done, info = env.step(a)
        s = ns
        env.render()
        if done:
            break
        
    env.close()