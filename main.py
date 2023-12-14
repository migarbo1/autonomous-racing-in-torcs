from ppo.ppo import PPO
from torcs_python_client.torcs_env import TorcsEnv
import torch
import os

def training_finished_procedure(env: TorcsEnv, model: PPO):
    env.training_data['total_training_timesteps'] += model.current_timesteps
    env.save_training_data()
    torch.save(model.actor.state_dict(), './ppo_actor.pth')
    torch.save(model.critic.state_dict(), './ppo_critic.pth')
    os.system(f'pkill torcs')

if __name__ == '__main__':
    env = TorcsEnv()
    #TODO: make console parameter
    timesteps = 50000
    model = PPO(env)
    try:
        model.learn(timesteps)
    except:
        training_finished_procedure(env, model)
    training_finished_procedure(env, model)

    

