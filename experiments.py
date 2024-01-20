import torcs_python_client.snakeoil3_gym as snakeoil 
from torcs_python_client.torcs_env import TorcsEnv
from ppo.ppo import PPO
from pathlib import Path
import numpy as np
import torch
import os

def change_track(track, prev_track):
    with open(Path.home() / '.torcs/config/raceman/quickrace.xml', 'r') as file:
        content = file.read()

    open(Path.home() / '.torcs/config/raceman/quickrace.xml', 'w').close() #reset file contents

    with open(Path.home() / '.torcs/config/raceman/quickrace.xml', 'w') as file:
        content = content.replace(f'<attstr name="name" val="{prev_track}"/>', f'<attstr name="name" val="{track}"/>')
        file.write(content)

def compute_reward(states):
    total_reward = 0
    for state in states:
        total_reward += (state['speedX'] * np.cos(state['angle']) - abs(state['speedX']*np.sin(state['angle'])) - abs(2*state['speedX']*np.sin(state['angle'])*state['trackPos']) - state['speedY']*np.cos(state['angle']))
    return total_reward

def write_results(results):
    with open(f'{os.getcwd()}/results.txt', 'w') as file:
        total_text = ''
        for k in results.keys():
            total_text += f'{k}: {results[k]}\n'
        file.write(total_text)

if __name__ == '__main__':
    torch.set_default_device('cuda')
    snakeoil.set_textmode(False)
    snakeoil.set_tracks(track_list=['quickrace'])
    tracks = ['brondehach','g-track-1', 'forza', 'g-track-2', 'g-track-3', 'ole-road-1', 'ruudskogen', 'street-1', 'wheel-1', 'wheel-2', 'aalborg', 'alpine-1', 'alpine-2', 'e-track-1', 'e-track-2', 'e-track-4', 'e-track-6', 'eroad', 'e-track-3']
    results= {}
    prev_track = tracks[0]
    for track in tracks:
        change_track(track, prev_track)
        rollout_distances = []
        rollout_rewards = []
        avg_speeds = []
        max_speeds = []
        for i in range(5):
            try:
                env = TorcsEnv()
                model = PPO(env, test=True)
                model.eval_max_timesteps = 500000
                model.launch_eval()
                env.kill_torcs()
                rollout_distances.append(model.env.training_data['eval_results'][-1]['dist_raced'])
                
                r = compute_reward(model.env.training_data['eval_results'][-1]['observation_rollout'])
                rollout_rewards.append(r)
                
                avg_speeds.append(model.env.training_data['eval_results'][-1]['avg_speed'])
                max_speeds.append(model.env.training_data['eval_results'][-1]['max_speed'])
            except Exception as e:
                print(e)
        prev_track = track
        results[track] = '{' + f'max_rollout_dist: {np.max(rollout_distances):4f}, avg_rollout_dist: {np.mean(rollout_distances):4f}, min_rollout_dist: {np.min(rollout_distances):4f}, max_score: {np.max(rollout_rewards):4f}, avg_score: {np.mean(rollout_rewards):4f}, avg_max_speed: {np.mean(max_speeds):4f}, avg_avg_speed: {np.mean(avg_speeds):4f}' + '}'
    write_results(results)
    change_track('brondehach', tracks[-1])