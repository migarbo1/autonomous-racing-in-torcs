#imports
from snakeoil3_gym import PI
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import logging
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

        # Action order:[Accel, Brake, Steering]  
        action_lows = np.array([0.0, 0.0, -1.0])
        action_highs = np.array([1.0, 1.0, 1.0])
        self.action_space = spaces.Box(low=action_lows, high=action_highs)

       # Observation order:[Angle, focus(5), speedX, speedY, speedZ, track(19), trackPos]  
        observation_lows = np.array([-PI, -1, -np.inf, -np.inf, -np.inf, -1, np.inf], dtype='float')
        observation_highs = np.array([PI, 200, np.inf, np.inf, np.inf, 200, np.inf], dtype='float')
        self.observation_space = spaces.Box(low=observation_lows, high=observation_highs, shape=(1, 5, 1, 1, 1, 19, 1))

        self.client = None
        self.time_step = 0


    def step(self, actions: dict):
        # to think: how to smoothen steering movements -> if now it is -1 and action tells 0... ¿place in value function?  
        client = self.client

        driver_action = client.R.d

        for k, v in actions.items():
            driver_action[k] = v
        
        # set automatic gear
        driver_action['gear'] = 1 