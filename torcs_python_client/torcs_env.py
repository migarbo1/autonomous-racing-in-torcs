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

    def __init__(self, create_client = False, num_frames=5) -> None:
        # snakeoil.restart_torcs()

        # Action order:[Accel&Brake, steer]  
        action_lows = np.array([-1.0, -1.0])
        action_highs = np.array([1.0, 1.0])
        self.action_space = spaces.Box(low=action_lows, high=action_highs)

        self.num_frames = num_frames

        # Observation order:[Angle, speedX, speedY, speedZ, track(19), trackPos]  
        self.single_obs_lows = [-PI, -2**62, -2**62, -2**62, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2**62]
        self.single_obs_highs = [PI, 2**62, 2**62, 2**62, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 2**62]
        observation_lows = np.array(num_frames*self.single_obs_lows, dtype='float')
        observation_highs = np.array(num_frames*self.single_obs_highs, dtype='float')
        self.observation_space = spaces.Box(low=observation_lows, high=observation_highs)

        self.time_step = 0
        self.max_speed = 300.
        self.previous_state = None
        self.frame_stacking = [np.zeros(len(self.single_obs_lows),) for _ in range(self.num_frames)]

        self.has_moved = False
        self.min_speed = 3/self.max_speed

        self.training_data = self.load_training_data()

        # self.vision_angles = [-90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 75, 90]
        # self.vision_angles = [-45, -19, -12, -7, -4, -2.5, -1.7, -1, -.5, 0, .5, 1, 1.7, 2.5, 4, 7, 12, 19, 45]
        # self.vision_angles = [-45, -32, -24, -12, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 12, 24, 32, 45]
        self.vision_angles = [-45, -32, -23, -11, -7, -4, -2.8, -1.5, -.5, 0, .5, 1.5, 2.8, 4, 7, 11, 23, 32, 45]
    
        self.last_lap_time = 0
        self.client = snakeoil.Client(p=3001) if create_client else None


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
            getattr(self.observation, 'speedX') < self.min_speed and self.has_moved or \
            self.client.S.d['damage'] > 0:

            reward = -10000
            done = True
            self.client.R.d['meta'] = True
            self.client.respond_to_server()
        
        self.has_moved = self.has_moved or getattr(self.observation, 'speedX') > (self.min_speed*5)

        self.time_step += 1

        self.previous_state = self.client.S.d.copy() # store full state instead of "observation" so we can track all values

        # print('RECEIVED OBSERVATION:', self.observation)

        obs_array = self.observation2array(self.observation)
        self.append_state_to_stack(obs_array)

        complete_obs = self.get_complete_obs()

        return np.array(complete_obs), reward, done, None, None # last Nones to compy with gym syntax

    
    def reset(self, eval = False):
        self.time_step = 0
        self.has_moved = False
        self.previous_state = None
        self.frame_stacking = [np.zeros(len(self.single_obs_lows),) for _ in range(self.num_frames)]

        snakeoil.restart_torcs(eval)
        
        self.client = snakeoil.Client(p=3001)
        self.client.MAX_STEPS = np.inf

        self.observation = self.client.get_servers_input()
        obs_array = self.observation2array(self.parse_torcs_input(self.observation))
        self.append_state_to_stack(obs_array)
        complete_obs = self.get_complete_obs()

        return np.array(complete_obs), None # to comply with Gym standard


    def get_complete_obs(self):
        # res = []

        # for i in range(len(self.frame_stacking[-1])):
        #     res.append(self.frame_stacking[-1][i])
        #     res.append(self.frame_stacking[-1][i] - self.frame_stacking[0][i])
        
        # return res
        return np.array(self.frame_stacking).flatten()


    def append_state_to_stack(self, state):
        aux = self.frame_stacking[1:]
        aux.append(state)
        self.frame_stacking = aux.copy()


    def compute_reward(self, state):
        # get speed of first frame of stack
        prev_speed = self.frame_stacking[0][1]
        speed_x = state['speedX']/self.max_speed
        speed_y = state['speedY']/self.max_speed
        angle = state['angle']/PI
        speed_dif = abs(speed_x - prev_speed)
        speed_reward = 2*speed_dif + speed_x * (np.cos(angle) - np.sin(abs(angle))) - abs(speed_y)*np.cos(angle)
        
        # get angle of first frame of stack
        prev_angle = self.frame_stacking[0][0]
        angle_variation = 5*abs(angle - prev_angle) # TODO: el problema del waving puede ser que mira el t-5 y no el numero de veces que cambia el angulo entre t-5 y t


        reward = speed_reward - angle_variation#- abs(state['trackPos']) #- angle_norm
            # + forward_view

        if state['lastLapTime'] > 0 and state['lastLapTime'] != self.last_lap_time:
            reward = reward + 100
            self.last_lap_time = state['lastLapTime']

        # print('SPEED REWARD: ', speed_reward)
        # print('STEER REWARD: ', steer_reward)
        print(f'sp_reward: {speed_reward:4f}; delta_sp: {speed_dif:4f}; delta_angle: {angle_variation:4f}; Reward: {reward:4f}')

        return reward


    def compute_gear(self, speed):
        gear = 1
        if speed > 120:
            gear = 2
        if speed > 140:
            gear = 3
        if speed > 190:
            gear = 4
        if speed > 240:
            gear = 5
        if speed > 270:
            gear = 6
        if speed > 295:
            gear = 7
        # if speed > 55:
        #     gear = 2
        # if speed > 90:
        #     gear = 3
        # if speed > 130:
        #     gear = 4
        # if speed > 165:
        #     gear = 5
        return gear


    def observation2array(self, observation):
        res = []
        res.append(getattr(observation,'angle'))
        res.append(getattr(observation, 'speedX'))
        res.append(getattr(observation, 'speedY'))
        res.append(getattr(observation, 'speedZ'))
        res = res + list(getattr(observation, 'track'))
        res.append(getattr(observation, 'trackPos'))
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
