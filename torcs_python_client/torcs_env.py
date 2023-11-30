#imports
from . import snakeoil3_gym as snakeoil
from gymnasium import spaces
import numpy as np
import collections
import logging
import random
import time
import os

PI = snakeoil.PI

def restart_torcs(mode):
    print(os.getcwd())
    logging.log(0, 'Killing torcs and re-launching...')
    os.system(f'pkill torcs')
    time.sleep(0.5)
    os.system('torcs -nofuel -nodamage &')
    time.sleep(0.5)
    os.system(f'sh {os.getcwd()}/torcs_python_client/autostart_{mode}.sh')
    time.sleep(0.5)


class TorcsEnv:

    def __init__(self, create_client = False) -> None:
        restart_torcs(random.sample(['race', 'practice'], 1)[0])

        # Action order:[Accel, Brake, Steering]  
        action_lows = np.array([0.0, 0.0, -1.0])
        action_highs = np.array([1.0, 1.0, 1.0])
        self.action_space = spaces.Box(low=action_lows, high=action_highs)

        # Observation order:[Angle, speedX, speedY, speedZ, track(19), trackPos]  
        observation_lows = np.array([-PI, -2**62, -2**62, -2**62, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2**62], dtype='float')
        observation_highs = np.array([PI, 2**62, 2**62, 2**62, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 2**62], dtype='float')
        self.observation_space = spaces.Box(low=observation_lows, high=observation_highs)


        self.client = snakeoil.Client(p=3001) if create_client else None
        self.time_step = 0
        self.max_speed = 330.
        self.previous_state = None


    def step(self, actions: dict):
        # to think: how to smoothen steering movements -> if now it is -1 and action tells 0... ¿place in value function?  
        done = False

        if isinstance(actions, np.ndarray):
            actions = self.action_array2dict(actions)

        # set action dict to the selected actions by network
        for k, v in actions.items():
            self.client.R.d[k] = v
        
        # set automatic gear
        self.client.R.d['gear'] = self.compute_gear(self.client.S.d['speedX'])

        # Apply the Agent's action into torcs
        self.client.respond_to_server()
        
        # Get the response of TORCS
        self.client.get_servers_input()
        self.observation = self.parse_torcs_input(self.client.S.d)
        
        #TODO:
        reward = self.compute_reward(self.client.S.d)

        # episode termination conditions: out of track or running backwards ¿small progress?
        if min(getattr(self.observation, 'track')) < 0 or \
            np.cos(getattr(self.observation, 'angle')) < 0:
            reward = -1
            done = True
            self.client.R.d['meta'] = True
            self.client.respond_to_server()
        
        self.time_step += 1

        return self.observation2array(self.observation), reward, done, None, None # last Nones to compy with gym syntax

    
    def reset(self):
        self.time_step = 0

        if self.client:
            self.client.R.d['meta'] = 1
            self.client.respond_to_server()

            # TO avoid memory leak re-launch torcs from time to time
            if random.random() < 0.33:
               restart_torcs(random.sample(['race', 'practice'], 1)[0])
        
        self.client = snakeoil.Client(p=3001)
        self.client.MAX_STEPS = np.inf

        self.observation = self.client.get_servers_input()
        print('reset:', self.observation)

        return self.observation2array(self.parse_torcs_input(self.observation)), None # to comply with Gym standard


    def compute_reward(self, state):
        prev_speed = self.previous_state['speedX'] if self.previous_state != None else 0
        reward = abs(state['speedX'] - prev_speed) + state['speedX']/self.max_speed
        self.previous_state = state
        return reward


    def compute_gear(self, speed):
        gear = 1
        if speed > 115:
            gear = 2
        if speed > 140:
            gear = 3
        if speed > 190:
            gear = 4
        if speed > 240:
            gear = 5
        if speed > 270:
            gear = 6
        if speed > 300:
            gear = 7
        return gear


    def observation2array(self, observation):
        res = []
        res.append(getattr(observation,'angle'))
        res.append(getattr(observation, 'speedX'))
        res.append(getattr(observation, 'speedY'))
        res.append(getattr(observation, 'speedZ'))
        res = res + list(getattr(observation, 'track'))
        res.append(getattr(observation, 'trackPos'))
        print(res)
        return np.array(res)


    def action_array2dict(self, actions):
        res = {
            'accel': actions[0],
            'brake': actions[1],
            'steering': actions[2]
            }
        return res


    def kill_torcs(self):
        os.system(f'pkill torcs')


    def parse_torcs_input(self, obs_dict: dict):
        keys = ['angle', 'speedX', 'speedY', 'speedZ', 'track', 'trackPos']
        observation = collections.namedtuple('observation', keys)
        return observation(
            angle=np.array(obs_dict['angle'], dtype=np.float32)/PI,
            speedX=np.array(obs_dict['speedX'], dtype=np.float32)/self.max_speed,
            speedY=np.array(obs_dict['speedY'], dtype=np.float32)/self.max_speed,
            speedZ=np.array(obs_dict['speedZ'], dtype=np.float32)/self.max_speed,
            track=np.array(obs_dict['track'], dtype=np.float32)/200.,
            trackPos=np.array(obs_dict['trackPos'], dtype=np.float32)
        )