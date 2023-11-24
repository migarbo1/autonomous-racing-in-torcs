import gymnasium as gym
from ppo.ppo import PPO
from torcs_python_client.torcs_env import TorcsEnv

if __name__ == '__main__':
    env = TorcsEnv()
    model = PPO(env)
    model.learn(100000)
