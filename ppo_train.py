import torch
import datetime
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from env import create_env
from config import ppo_pong_config
from models.ppo import PPO
from collections import deque


if __name__ == '__main__':
    env_list = [create_env(ppo_pong_config) for _ in range(ppo_pong_config.num_env)]
    agent = PPO(env_list[0], ppo_pong_config)
    agent.set_optimizer()
    assert ppo_pong_config.batch_size % ppo_pong_config.num_env == 0

    dt_now = datetime.datetime.now()
    logdir = f"logdir/{dt_now.strftime('%y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(logdir)

    score_que = deque([], maxlen=ppo_pong_config.num_eval_episode)
    count_step_que = deque([], maxlen=ppo_pong_config.num_eval_episode)

    score = 0
    count_step = 0
    s_list = [env.reset() for env in env_list]
    score_list = [0 for env in env_list]
    count_step_list = [0 for env in env_list]

    num_iteration = int(ppo_pong_config.train_env_steps / ppo_pong_config.num_env / ppo_pong_config.seq_length)
    for step_iteration in range(num_iteration):
        for i_env in range(ppo_pong_config.num_env):
            env = env_list[i_env]
            s = s_list[i_env]
            for _ in range(ppo_pong_config.seq_length):
                pi, a = agent.action(s)
                s_next, r, done, info = env.step(a)
                agent.add_to_batch(s, pi, a, r, s_next, done)
                s = s_next
                score_list[i_env] += r
                count_step_list[i_env] += 1

                s_list[i_env] = s
                if done:
                    s = env.reset()
                    s_list[i_env] = s

                    score_que.append(score_list[i_env])
                    count_step_que.append(count_step_list[i_env])

                    score_list[i_env] = 0
                    count_step_list[i_env] = 0

                    break
         
        if len(agent.batch) == ppo_pong_config.batch_size:
            loss_critic_avg, entropy_avg = agent.train()
            writer.add_scalar('Train/loss_critic', loss_critic_avg, step_iteration)
            writer.add_scalar('Train/entropy', entropy_avg, step_iteration)

        if len(score_que) == ppo_pong_config.num_eval_episode:
            score_avg = np.mean(score_que)
            count_step_avg = np.mean(count_step_que)
            writer.add_scalar('Env/score_avg', score_avg, step_iteration)
            writer.add_scalar('Env/count_step_avg', count_step_avg, step_iteration)
            
            print(
                f"[{step_iteration}] score_avg: {score_avg:.3f} "
                f"count_step_avg: {count_step_avg:.3f} "
                f"loss_critic_avg: {loss_critic_avg:.3f} "
                f"entropy_avg: {entropy_avg:.3f} "
            )
            score_que.clear()
            count_step_que.clear()

    torch.save(agent.state_dict(), f"{logdir}/state_dict.pth")