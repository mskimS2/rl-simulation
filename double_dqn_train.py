import torch
import datetime
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from env import create_env
from config import double_dqn_pong_config
from models.double_dqn import DoubleDQN
from utils import get_eps, eval_agent


if __name__ == "__main__":
    config = double_dqn_pong_config
    env = create_env(config)
    env_eval = create_env(config)
    agent = DoubleDQN(env, config)
    agent.set_optimizer()
    
    dt_now = datetime.datetime.now()
    logdir = f"logdir/{dt_now.strftime('%y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(logdir)

    # Reset Replay Buffer
    init_replay_buffer_size = int(config.replay_init_ratio * config.replay_capacity)
    s = env.reset()
    step_count = 0
    for _ in range(init_replay_buffer_size):
        a = np.random.choice(env.action_space.n)  # uniform random action
        s_next, r, done, info = env.step(a)
        step_count += 1

        transition = (s, a, r, s_next, done)
        agent.replay_memory.append(transition)

        s = s_next
        if done:
            s = env.reset()
            step_count = 0

    # Train agent
    s = env.reset()
    step_count = 0
    for step_train in range(config.train_env_steps):
        eps = get_eps(config, step_train)
        is_random_action = np.random.choice(2, p=[1 - eps, eps])
        if is_random_action:
            a = np.random.choice(env.action_space.n)  # uniform random action
        else:
            a = agent.get_argmax_action(s)
        
        s_next, r, done, info = env.step(a)
        step_count += 1

        transition = (s, a, r, s_next, done)
        agent.replay_memory.append(transition)

        s = s_next
        if done:
            s = env.reset()
            step_count = 0

        if step_train % config.target_update_period == 0:
            agent.update_target_network()

        if step_train % 4 == 0:
            loss = agent.train()

        if step_train % config.eval_period == 0:
            score_avg, step_count_avg = eval_agent(config, env_eval, agent)
            print(
                f"[{step_train}] eps: {eps:.3f} loss: {loss:.3f} "
                + f"score_avg: {score_avg:.3f} step_count_avg: {step_count_avg:.3f}"
            )
            writer.add_scalar("Train/loss", loss, step_train)
            writer.add_scalar("Eval/score_avg", score_avg, step_train)
            writer.add_scalar("Eval/step_count_avg", step_count_avg, step_train)

    torch.save(agent.state_dict(), f"{logdir}/dqn.pth")