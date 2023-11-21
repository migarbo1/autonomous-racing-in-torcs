#imports
import snakeoil3_gym as snakeoil
from snakeoil3_gym import PI
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import collections
import logging
import random
import time
import os


def restart_torcs():
    logging.log(0, 'Killing torcs and re-launching...')
    os.system(f'pkill torcs')
    time.sleep(0.5)
    os.system('torcs -nofuel -nodamage &')
    time.sleep(0.5)
    os.system('sh autostart.sh')
    time.sleep(0.5)


class TorcsEnv:

    def __init__(self) -> None:

        restart_torcs()

    #     # Action order:[Accel, Brake, Steering]  
    #     action_lows = np.array([0.0, 0.0, -1.0])
    #     action_highs = np.array([1.0, 1.0, 1.0])
    #     self.action_space = spaces.Box(low=action_lows, high=action_highs)

    #    # Observation order:[Angle, focus(5), speedX, speedY, speedZ, track(19), trackPos]  
    #     observation_lows = np.array([-PI, -1, -np.inf, -np.inf, -np.inf, -1, np.inf], dtype='float')
    #     observation_highs = np.array([PI, 200, np.inf, np.inf, np.inf, 200, np.inf], dtype='float')
    #     self.observation_space = spaces.Box(low=observation_lows, high=observation_highs, shape=(1, 5, 1, 1, 1, 19, 1))

        self.client = snakeoil.Client(p=3001)
        self.time_step = 0
        self.max_speed = 330.


    def step(self, actions: dict):
        # to think: how to smoothen steering movements -> if now it is -1 and action tells 0... ¿place in value function?  
        done = False

        # set action dict to the selected actions by network
        for k, v in actions.items():
            self.client.R.d[k] = v
        
        # set automatic gear
        self.client.R.d['gear'] = 1 

        # Apply the Agent's action into torcs
        self.client.respond_to_server()

        # Get the response of TORCS
        self.client.get_servers_input()

        self.observation = self.parse_torcs_input(self.client.S.d)
        
        #TODO:
        reward = 0

        # episode termination conditions: out of track or running backwards ¿small progress?
        if min(self.observation['track']) < 0 or \
            np.cos(self.observation['angle']) < 0:
            reward = -1
            done = True
            self.client.R.d['meta'] = True
            self.client.respond_to_server()
        
        self.time_step += 1

        return self.observation, reward, done, None # last item to compy with gym syntax

    
    def reset(self):
        self.time_step = 0

        self.client.R.d['meta'] = True
        self.client.respond_to_server()

        # TO avoid memory leak re-launch torcs from time to time
        if random.uniform(0,1) < 0.33:
            restart_torcs()

        self.client = snakeoil.Client(p=3001)
        self.client.MAX_STEPS = np.inf

        self.observation = self.client.get_servers_input()

        return self.observation


    def kill_torcs(self):
        os.system(f'pkill torcs')


    def parse_torcs_input(self, obs_dict: dict):
        keys = ['angle', 'focus', 'speedX', 'speedY', 'speedZ', 'track, trackPos']
        observation = collections.namedtuple('observation', keys)
        return observation(
            focus=np.array(obs_dict['focus'], dtype=np.float32)/200.,
            speedX=np.array(obs_dict['speedX'], dtype=np.float32)/self.max_speed,
            speedY=np.array(obs_dict['speedY'], dtype=np.float32)/self.max_speed,
            speedZ=np.array(obs_dict['speedZ'], dtype=np.float32)/self.max_speed,
            track=np.array(obs_dict['track'], dtype=np.float32)/200.,
            trackPos=np.array(obs_dict['trackPos'], dtype=np.float32),
        )