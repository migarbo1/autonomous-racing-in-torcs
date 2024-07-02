import torcs_python_client.snakeoil3_gym as snakeoil 
from torcs_python_client.torcs_env import TorcsEnv
from ppo.ppo_lstm import PPOLSTM
import matplotlib.pyplot as plt
from pathlib import Path
from ppo.ppo import PPO
import numpy as np
import torch
import click
import time
import math
import sys
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
@click.option('--num_tries', '-l', default=3, help='Number of tries to select best')
@click.option('--model_name', '-n', default='./weights/ppo', help='The name of the model file')
@click.option('--track', '-t', default='wheel-2', help='Track to obtain telemetry comparison')
@click.option('--visual_mode', '-v', is_flag=True, default=False, help='show torcs visual interface')
@click.option('--output_file', default='results', help='The name of the results file')
def main(num_tries, model_name, track, visual_mode, output_file):
    print(num_tries, model_name, track, visual_mode, output_file)
    torch.set_default_device('cuda')
    textmode = not visual_mode
    snakeoil.set_textmode(textmode)
    snakeoil.set_tracks(track_list=['quickrace'])
    prev_track = 'brondehach'
    change_track(track, prev_track)
    max_speed = 0
    colors=['b', 'c', 'm']
    for idx, m_name in enumerate(model_name.split(',')):
        env = TorcsEnv(
            num_frames=5 if m_name.__contains__('ppo_FS5') else 1, 
            load_tr_data=False,
            join_accel_brake= not m_name.__contains__('M4')
        )
        model = PPO(
            env, 
            test=True,
            model_name=m_name
        )
        model.eval_max_timesteps = 50000
        rollout_distances = []
        observations = {}
        for i in range(num_tries):
            print(f'rollout {i+1} of {num_tries}')
            try:
                res = model.launch_eval(only_practice=False)
                env.kill_torcs()
                rollout_distances.append(res['dist_raced'])
                observations[i] = res['observation_rollout']
                
            except Exception as e:
                with open(f'{os.getcwd()}/errlog.txt', 'w') as file:
                    file.write(e)

        # select rollout with maximum distance covered
        best_try = np.argmax(rollout_distances)
        best_data = observations[best_try]  
        
        # remove first lap
        best_data = list(filter( lambda x: x['lastLapTime'] != None and x['lastLapTime'] > 0, best_data))

        # find observation next to the end of the best lap and trim from there onwards
        end_best_lap = np.argmin([obs['lastLapTime'] for obs in best_data])
        best_lap_time = best_data[end_best_lap]['lastLapTime']
        best_lap_data = best_data[:end_best_lap]

        # get the time of the previous lap at that point    
        prev_lap_time = best_lap_data[-1]['lastLapTime']

        # find the start of the best lap by finding the first occurrence of the last lap time
        start_best_lap = np.argmax(list(int(obs['lastLapTime'] == prev_lap_time) for obs in best_lap_data))
        best_lap_data = best_lap_data[start_best_lap:]


        speed_telemetry = [obs['speedX'] for obs in best_lap_data]
        distance_telemetry = [obs['distFromStart'] for obs in best_lap_data]
        plt.plot(distance_telemetry, speed_telemetry, c=colors[idx], label=f"{m_name.replace('./ictai_exp/ppo_', '')}: {best_lap_time}s")
        max_speed = max(np.max(speed_telemetry), max_speed)
    
    plt.title(f"Telemetry on {track}")
    plt.xlabel("Distance from goal line (m)")
    plt.ylabel('Speed (km/h)')
    if track == 'wheel-2':
        turns = [390, 575, 840, 970, 1150, 1375, 1550, 2150, 2290, 2650, 2800, 3350, 3765, 3965, 4950, 5400, 5600, 5700]
        plt.vlines(x=turns, colors='gray', linestyles='dotted', ymin = 0, ymax= max_speed)
    plt.legend()
    plt.show()
    # write_results(results, output_file)
    change_track('brondehach', track)

if __name__ == '__main__':
    main()
