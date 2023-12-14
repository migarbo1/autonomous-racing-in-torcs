#imports
import pickle
from . import snakeoil3_gym as snakeoil
from gymnasium import spaces
import numpy as np
import collections
import logging
import random
import time
import json
import math
import os

PI = snakeoil.PI

class TorcsEnv:

    def __init__(self, create_client = False) -> None:
        snakeoil.restart_torcs()

        # Action order:[Accel, Brake, steer]  
        action_lows = np.array([-1.0, -1.0])
        action_highs = np.array([1.0, 1.0])
        self.action_space = spaces.Box(low=action_lows, high=action_highs)

        # Observation order:[Angle, speedX, speedY, speedZ, track(19), trackPos, accelX, accelY]  
        observation_lows = np.array([-PI, -2**62, -2**62, -2**62, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2**62, -2**62, -2**62], dtype='float')
        observation_highs = np.array([PI, 2**62, 2**62, 2**62, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 2**62, 2**62, 2**62], dtype='float')
        self.observation_space = spaces.Box(low=observation_lows, high=observation_highs)

        self.client = snakeoil.Client(p=3001) if create_client else None
        self.time_step = 0
        self.max_speed = 330.
        self.previous_state = None

        self.has_moved = False
        self.min_speed = 3/self.max_speed

        self.training_data = self.load_training_data()


    def load_training_data(self):
        if os.path.isfile(f'{os.getcwd()}/training_data.pickle'):
            with open(f'{os.getcwd()}/training_data.pickle', 'rb') as file:
                return pickle.load(file)
        else:
            return {"total_training_timesteps": 0.0, "actor_episodic_avg_loss": [], "critic_episodic_avg_loss": [], "eval_results": []}


    def save_training_data(self):
        with open(f'{os.getcwd()}/training_data.pickle', 'wb') as file:
            pickle.dump(self.training_data, file, protocol=pickle.HIGHEST_PROTOCOL)


    def step(self, actions: list):
        # to think: how to smoothen steer movements -> if now it is -1 and action tells 0... ¿place in value function?  
        done = False

        if isinstance(actions, np.ndarray):
            actions = self.action_array2dict(actions)

        # print('Parsed actions:', actions)

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
            np.cos(getattr(self.observation, 'angle')) < 0 or \
            getattr(self.observation, 'speedX') < self.min_speed and self.has_moved:
            reward = -1000
            done = True
            self.client.R.d['meta'] = True
            self.client.respond_to_server()
        
        self.has_moved = self.has_moved or getattr(self.observation, 'speedX') > (self.min_speed*2)

        self.time_step += 1

        self.previous_state = self.client.S.d.copy() # store full state instead of "observation" so we can track all values

        # print('RECEIVED OBSERVATION:', self.observation)

        return self.observation2array(self.observation), reward, done, None, None # last Nones to compy with gym syntax

    
    def reset(self, eval = False):
        self.time_step = 0
        self.has_moved = False

        if self.client:
            self.client.R.d['meta'] = 1
            self.client.respond_to_server()

            # TO avoid memory leak re-launch torcs from time to time
            #if random.random() < 0.33:
            snakeoil.restart_torcs(eval)
        
        self.client = snakeoil.Client(p=3001)
        self.client.MAX_STEPS = np.inf

        self.observation = self.client.get_servers_input()

        return self.observation2array(self.parse_torcs_input(self.observation)), None # to comply with Gym standard


    def compute_reward(self, state):
        prev_speed = self.previous_state['speedX'] if self.previous_state != None else 0
        speed_reward = abs(state['speedX'] - prev_speed)

        prev_angle = self.previous_state['angle'] if self.previous_state != None else 0 
        angle_variation = abs(state['angle'] - prev_angle)
        # steer_reward = -10 * (abs(np.sin(state['angle'])) + angle_variation)
        
        # reward = speed_reward - abs(np.sin(state['angle']))#+ steer_reward

        forward_view =(np.mean(state['track'][8:11])/200)

        vertical_speed = state['speedX'] * abs(np.cos(state['angle']))
        horizontal_speed = state['speedX'] * abs(np.sin(state['angle']))

        reward = speed_reward + (vertical_speed + horizontal_speed )/self.max_speed \
            - angle_variation
            # + forward_view

        # print('SPEED REWARD: ', speed_reward)
        # print('STEER REWARD: ', steer_reward)
        print(f'v_sp: {vertical_speed/self.max_speed:4f}; h_sp: {horizontal_speed/self.max_speed:4f}; delta: {angle_variation:4f}; REWARD: {reward:4f}')

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
        res.append(getattr(observation, 'accelX'))
        res.append(getattr(observation, 'accelY'))
        return np.array(res)


    def action_array2dict(self, actions):
        accel, brake = 0, 0

        if actions[0] > 0:
            accel = actions[0]
        else:
            brake = abs(actions[0])

        res = {
            'accel': accel,
            'brake': brake, 
            'steer': actions[1]
            }

        return res


    def kill_torcs(self):
        os.system(f'pkill torcs')


    def parse_torcs_input(self, obs_dict: dict):
        keys = ['angle', 'speedX', 'speedY', 'speedZ', 'track', 'trackPos', 'accelX', 'accelY']
        observation = collections.namedtuple('observation', keys)

        prev_speedx = self.previous_state['speedX'] if self.previous_state != None else 0
        prev_time = self.previous_state['curLapTime'] if self.previous_state != None else 0
        accelX = (obs_dict['speedX'] - prev_speedx) / (obs_dict['curLapTime'] - prev_time) if obs_dict['curLapTime'] - prev_time != 0 else 0
        accelY = (obs_dict['speedY'] - prev_speedx) / (obs_dict['curLapTime'] - prev_time) if obs_dict['curLapTime'] - prev_time != 0 else 0

        return observation(
            angle=np.array(obs_dict['angle'], dtype=np.float32)/PI,
            speedX=np.array(obs_dict['speedX'], dtype=np.float32)/self.max_speed,
            speedY=np.array(obs_dict['speedY'], dtype=np.float32)/self.max_speed,
            speedZ=np.array(obs_dict['speedZ'], dtype=np.float32)/self.max_speed,
            track=np.array(obs_dict['track'], dtype=np.float32)/200.,
            trackPos=np.array(obs_dict['trackPos'], dtype=np.float32),
            accelX=np.array(accelX, dtype=np.float32),
            accelY=np.array(accelY, dtype=np.float32)
        )
