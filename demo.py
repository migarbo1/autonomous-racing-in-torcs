import torcs_python_client.snakeoil3_gym as snakeoil 
from torcs_python_client.torcs_env import TorcsEnv
from ppo.ppo import PPO
import torch


if __name__ == '__main__':
    snakeoil.set_textmode(False)
    train_tracks = snakeoil.TRACKS.copy() + ['practice']
    for track in train_tracks:
        snakeoil.set_tracks(track_list=[track])
        env = TorcsEnv()
        model = PPO(env)
        model.eval_max_timesteps = 1000000
        for i in range(3):
            model.launch_eval()
    snakeoil.kill_torcs()