import torcs_python_client.snakeoil3_gym as snakeoil 
from torcs_python_client.torcs_env import TorcsEnv
from ppo.ppo import PPO
import numpy as np
import torch


if __name__ == '__main__':
    torch.set_default_device('cuda')
    snakeoil.set_textmode(False)
    train_tracks = snakeoil.TRACKS.copy() + ['practice']
    for track in train_tracks:
        rollout_distances = []
        snakeoil.set_tracks(track_list=[track])
        env = TorcsEnv()
        model = PPO(env, test=True)
        model.eval_max_timesteps = 500000
        for i in range(3):
            model.launch_eval(False)
            rollout_distances.append(model.env.training_data['eval_results'][-1]['dist_raced'])
        print(np.mean(rollout_distances))
        print(np.max(rollout_distances))
        print(np.min(rollout_distances))
    snakeoil.kill_torcs()