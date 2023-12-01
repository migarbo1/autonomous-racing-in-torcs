from ppo.ppo import PPO
from torcs_python_client.torcs_env import TorcsEnv
import torch
import os

if __name__ == '__main__':
    env = TorcsEnv()
    model = PPO(env)
    model.learn(400000)
    torch.save(model.actor.state_dict(), './ppo_actor.pth')
    torch.save(model.critic.state_dict(), './ppo_critic.pth')
    os.system(f'pkill torcs')

