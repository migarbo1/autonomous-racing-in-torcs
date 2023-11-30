import gymnasium as gym
from ppo.ppo import PPO
from torcs_python_client.torcs_env import TorcsEnv
import torch

if __name__ == '__main__':
    env = TorcsEnv()
    model = PPO(env)
    model.learn(100000)
    torch.save(model.actor.state_dict(), './ppo_actor.pth')
    torch.save(model.critic.state_dict(), './ppo_critic.pth')

