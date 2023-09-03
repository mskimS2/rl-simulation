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


config = AttrDict(
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

