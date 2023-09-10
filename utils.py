import torch
import random
import numpy as np
from collections import deque


class ReplayMemory:
    def __init__(self, config):
        self.config = config
        self.buffer = deque([], maxlen=self.config.replay_capacity)

    def getsize(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, size):
        if len(self.buffer) < size:
            assert False, f"Buffer size ({len(self.buffer)}) is smaller than the sample size ({size})"
            
        samples = random.sample(self.buffer, size)
        return samples


def get_eps(config, step):
    eps_init = config.eps_init
    eps_final = config.eps_final
    if step >= config.eps_decrease_step:
        eps = eps_final
    else:
        m = (eps_final - eps_init) / config.eps_decrease_step
        eps = eps_init + m * step
    return eps



def eval_agent(config, env, agent):
    score_sum = 0
    step_count_sum = 0
    for _ in range(config.num_eval_episode):
        s = env.reset()
        step_count = 0
        done = False
        score = 0
        while not done:
            with torch.no_grad():
                a = agent.get_argmax_action(s)

            s_next, r, done, info = env.step(a)
            step_count += 1
            
            score += r
            s = s_next
        
        score_sum += score
        step_count_sum += step_count

    score_avg = score_sum / config.num_eval_episode
    step_count_avg = step_count_sum / config.num_eval_episode
    return score_avg, step_count_avg


def set_randomness(random_seed: int = 2023):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False