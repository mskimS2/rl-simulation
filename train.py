import gym
import torch
import numpy as np

from models.layers.actor import Actor
from models.layers.critic import Critic
from models.ddpg import DDPG, OrnsteinUhlenbeckProcess
from utils.memory import ReplayMemory
from config import ddpg_pendulum_config as args
from utils.train_utils import to_tensor, prepare_training_inputs
from utils.update_net import soft_update


if __name__ == '__main__':    
    actor, actor_target = Actor(), Actor()
    critic, critic_target = Critic(), Critic()
    agent = DDPG(
        critic=critic,
        critic_target=critic_target,
        actor=actor,
        actor_target=actor_target).to(args.device,
    )

    memory = ReplayMemory(args.memory_size)

    env = gym.make("Pendulum-v1")
    for n_epi in range(args.total_eps):
        ou_noise = OrnsteinUhlenbeckProcess(mu=np.zeros(1))
        s = env.reset()
        cum_r = 0

        while True:
            s = to_tensor(s, size=(1, 3)).to(args.device)
            a = agent.get_action(s).cpu().numpy() + ou_noise()[0]
            ns, r, done, info = env.step(a)

            experience = (
                s,
                torch.tensor(a).view(1, 1),                          
                torch.tensor(r).view(1, 1),
                torch.tensor(ns).view(1, 3),
                torch.tensor(done).view(1, 1),
            )
            memory.push(experience)

            s = ns
            cum_r += r

            if len(memory) >= args.sampling_only_until:
                # train agent
                sampled_exps = memory.sample(args.batch_size)
                sampled_exps = prepare_training_inputs(sampled_exps, device=args.device)
                agent.update(*sampled_exps)
                # update target networks
                soft_update(agent.actor, agent.actor_target, args.tau)
                soft_update(agent.critic, agent.critic_target, args.tau)        

            if done:
                break

        if n_epi % args.print_every == 0:
            msg = (n_epi, cum_r) # ~ -100 cumulative reward = "solved"
            print("Episode : {} | Cumulative Reward : {} |".format(*msg))

    torch.save(agent.state_dict(), "ddpg_pendulum_v1.pth")
    
    
