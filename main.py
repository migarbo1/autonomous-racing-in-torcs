from ppo.ppo_lstm import PPOLSTM
from ppo.ppo import PPO
from torcs_python_client.torcs_env import TorcsEnv
import torch
import sys
import os

def training_finished_procedure(env: TorcsEnv, model: PPOLSTM):
    env.training_data['total_training_timesteps'] += model.current_timesteps
    env.save_training_data()
    os.system(f'pkill torcs')

if __name__ == '__main__':
    torch.set_default_device('cuda')
    
    num_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    use_human_data = bool(sys.argv[2] == 'True') if len(sys.argv)>2 else False
    env = TorcsEnv(num_frames = num_frames)
    #TODO: make console parameter
    timesteps = 18000000
    model = PPO(env, use_human_data=use_human_data)
    try:
        model.learn(timesteps)
    except Exception as e:
        print(e)
        training_finished_procedure(env, model)
    training_finished_procedure(env, model)

    

