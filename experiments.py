import torcs_python_client.snakeoil3_gym as snakeoil 
from torcs_python_client.torcs_env import TorcsEnv
from ppo.ppo_lstm import PPOLSTM
from ppo.ppo import PPO
from pathlib import Path
import numpy as np
import torch
import click
import math
import os

def change_track(track, prev_track):
    config_path = Path.home() / '.torcs/config/raceman' if not snakeoil.TEXTMODE else 'configs'
    with open(f'{config_path}/quickrace.xml', 'r') as file:
        content = file.read()

    open(f'{config_path}/quickrace.xml', 'w').close() #reset file contents

    with open(f'{config_path}/quickrace.xml', 'w') as file:
        content = content.replace(f'<attstr name="name" val="{prev_track}"/>', f'<attstr name="name" val="{track}"/>')
        file.write(content)

def write_results(results, output_file):
    with open(f'{os.getcwd()}/{output_file}.txt', 'w') as file:
        total_text = ''
        for k in results.keys():
            total_text += f'{k}: {results[k]}\n'
        file.write(total_text)

@click.command()
@click.option('--num_frames', '-f', default=1, help='Number of frames to be stacked in an observation')
@click.option('--timesteps', '-t', default=50000, help='Number of training steps')
@click.option('--join_accel_brake', '-j', is_flag=True, default=False, help='Combine accel and brake accions into a single one')
@click.option('--model_name', '-n', default='./weights/ppo', help='The name of the model file')
@click.option('--focus', is_flag=True, default=False, help='Narrows the view field')
@click.option('--visual_mode', '-v', is_flag=True, default=False, help='show torcs visual interface')
@click.option('--output_file', default='results', help='The name of the results file')
@click.option('--min_variance', default=0.05, help='Minimum value for exploration factor for mean sampling')
def main(num_frames, timesteps, join_accel_brake, model_name, focus, visual_mode, output_file, min_variance):
    print(num_frames, timesteps, join_accel_brake, model_name, focus, visual_mode, output_file, min_variance)
    torch.set_default_device('cuda')
    textmode = not visual_mode
    snakeoil.set_textmode(textmode)
    snakeoil.set_tracks(track_list=['quickrace'])
    tracks = ['brondehach','g-track-1', 'forza', 'g-track-2', 'g-track-3', 'ole-road-1', 'ruudskogen', 'street-1', 'wheel-1', 'wheel-2', 'aalborg', 'alpine-1', 'alpine-2', 'e-track-2', 'e-track-4', 'e-track-6', 'eroad', 'e-track-3', 'corkscrew'] #  'e-track-1',
    results= {}
    prev_track = tracks[0]
    for track in tracks:
        change_track(track, prev_track)
        print('current track: ', track)
        rollout_distances = []
        avg_speeds = []
        max_speeds = []
        env = TorcsEnv(
            num_frames=num_frames, 
            load_tr_data=False,
            join_accel_brake=join_accel_brake,
            focus=focus
        )
        model = PPO(
            env, 
            test=True,
            model_name=model_name,
            min_variance=min_variance
        )
        model.eval_max_timesteps = timesteps
        for i in range(7):
            print(f'rollout {i+1} of {7}')
            try:
                res = model.launch_eval(only_practice=False)
                env.kill_torcs()
                rollout_distances.append(res['dist_raced'])
                
                avg_speeds.append(res['avg_speed'])
                max_speeds.append(res['max_speed'])
            except Exception as e:
                with open(f'{os.getcwd()}/errlog.txt', 'w') as file:
                    file.write(e)
        prev_track = track
        rollout_distances = rollout_distances[1:]
        rollout_conf_int = 1.95 * (np.std(rollout_distances)/math.sqrt(len(rollout_distances)))
        results[track] = '{' + f'max_rollout_dist: {np.max(rollout_distances):4f}, avg_rollout_dist: {np.mean(rollout_distances):4f}, min_rollout_dist: {np.min(rollout_distances):4f}, rollout_conf_int: {rollout_conf_int:4f}, avg_max_speed: {np.mean(max_speeds):4f}, avg_avg_speed: {np.mean(avg_speeds):4f}' + '}'
    write_results(results, output_file)
    change_track('brondehach', tracks[-1])

if __name__ == '__main__':
    main()
