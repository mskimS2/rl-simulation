import gym
from wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv


def make_cartpole(env_name: str, num_envs: int, capture_video: bool, run_name: str):
    def thunk():
        if capture_video and num_envs == 0:
            env = gym.make(env_name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


if __name__ == "__main__":
    env = make_cartpole("CartPole-v1", 1, False, "test")()
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
        env.render()
    env.close()