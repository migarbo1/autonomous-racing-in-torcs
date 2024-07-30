from ppo.ppo_lstm import PPOLSTM
from ppo.ppo import PPO
from torcs_python_client.torcs_env import TorcsEnv
import torch
import click
import sys
import os

def training_finished_procedure(env: TorcsEnv, model: PPOLSTM):
    # env.training_data['total_training_timesteps'] += model.current_timesteps
    # env.save_training_data()
    os.system(f'pkill torcs')

@click.command()
@click.option('--num_frames', '-f', default=1, help='Number of frames to be stacked in an observation')
@click.option('--use_human_data', '-h', is_flag=True, default=False, help='Use buffer of human data')
@click.option('--timesteps', '-t', default=18000000, help='Number of training steps')
@click.option('--force_centerline', '-c', is_flag=True, default=False, help='Force the agent to learn to drive through the middle of the track. Only available for speed reward')
@click.option('--join_accel_brake', '-j', is_flag=True, default=False, help='Combine accel and brake accions into a single one')
@click.option('--model_name', '-n', default='./weights/ppo', help='The name of the model file')
@click.option('--hf', default='/human_data/formatted.pickle', help='The name of the model file')
@click.option('--variance', '-v', default=0.35, help='Exploration factor for mean sampling')
@click.option('--min_variance', default=0.05, help='Minimum value for exploration factor for mean sampling')
@click.option('--try_brake', is_flag=True, default=False, help='Exploration mechanism to force brake usage')
@click.option('--focus', is_flag=True, default=False, help='Narrows the view field')
@click.option('--lr', default=0.005, help='Leraning rate for the PPO algorithm')
@click.option('--hs', default=1920, help='number of human samples to introduce in batch')
@click.option('--reward', type=click.Choice(['speed', 'distance'], case_sensitive=False), default='speed')
def main(num_frames, use_human_data, timesteps, force_centerline, join_accel_brake, model_name, variance, try_brake, focus, lr, reward, hf, hs, min_variance):
    
    print(num_frames, use_human_data, timesteps, force_centerline, join_accel_brake, model_name, variance, try_brake, focus, lr, reward, hf, hs, min_variance)

    torch.set_default_device('cuda')
    env = TorcsEnv(
        num_frames=num_frames,
        force_centerline=force_centerline,
        join_accel_brake=join_accel_brake,
        speed_based_reward=(reward=='speed'),
        focus=focus
        )

    model = PPO(
        env, 
        use_human_data=use_human_data,
        human_data_file = hf,
        variance=variance,
        try_brake=try_brake,
        default_lr=lr,
        model_name=model_name,
        human_steps=hs,
        min_variance=min_variance
        )
    try:
        model.learn(timesteps)
    except Exception as e:
        print(f'ERROR: {e}')
        training_finished_procedure(env, model)
    training_finished_procedure(env, model)

if __name__ == '__main__':

    main()

    

