import torch

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)

    @classmethod
    def from_nested_dics(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key:cls.from_nested_dics(data[key]) for key in data})


dqn_pong_config = AttrDict(
    gamma=0.99,
    lr=5e-5,
    batch_size=64,
    hidden_size=128,
    replay_capacity=10000,
    replay_init_ratio=0.3,
    train_env_steps=200000,
    target_update_period=100,
    eps_init=1.0,
    eps_final=0.05,
    eps_decrease_step=100000,
    num_eval_episode=20,
    eval_period=500,
)

double_dqn_pong_config = AttrDict(
    gamma=0.99,
    lr=5e-5,
    batch_size=64,
    hidden_size=128,
    replay_capacity=10000,
    replay_init_ratio=0.3,
    train_env_steps=200000,
    target_update_period=500,
    eps_init=1.0,
    eps_final=0.05,
    eps_decrease_step=100000,
    num_eval_episode=20,
    eval_period=500,
)

ppo_pong_config = AttrDict(
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    k_epoch=8,
    lr=1e-4,
    c1=1,
    c2=0.5,
    c3=1e-3,
    num_env=8,
    seq_length=16,
    batch_size=64,
    minibatch_size=64,
    hidden_size=128,
    train_env_steps=1000000,
    num_eval_episode=100,
)

ddpg_pendulum_config = AttrDict(
    lr_actor=0.005,
    lr_critic=0.001,
    gamma=0.99,
    batch_size=256,
    memory_size=50000,
    tau=0.001, # polyak parameter for soft target update
    sampling_only_until=2000,
    device="cuda" if torch.cuda.is_available() else "cpu",
    total_eps=200,
    print_every=10,
)