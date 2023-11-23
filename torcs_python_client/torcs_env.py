#imports
import snakeoil3_gym as snakeoil
from snakeoil3_gym import PI
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
    os.system('sh autostart_race.sh')
    time.sleep(0.5)


class TorcsEnv:

    def __init__(self, create_client = False) -> None:

        restart_torcs()

        self.client = snakeoil.Client(p=3001) if create_client else None
        self.time_step = 0
        self.max_speed = 330.


    def step(self, actions: dict):
        # to think: how to smoothen steering movements -> if now it is -1 and action tells 0... ¿place in value function?  
        done = False

        # set action dict to the selected actions by network
        for k, v in actions.items():
            self.client.R.d[k] = v
        
        # set automatic gear
        self.client.R.d['gear'] = self.compute_gear(self.client.S.d['speedX'])

        # Apply the Agent's action into torcs
        self.client.respond_to_server()

        # Get the response of TORCS
        self.client.get_servers_input()
        print(self.client.S)
        self.observation = self.parse_torcs_input(self.client.S.d)
        
        #TODO:
        reward = 0

        # episode termination conditions: out of track or running backwards ¿small progress?
        if min(getattr(self.observation, 'track')) < 0 or \
            np.cos(getattr(self.observation, 'angle')) < 0:
            reward = -1
            done = True
            self.client.R.d['meta'] = True
            self.client.respond_to_server()
        
        self.time_step += 1

        return self.observation, reward, done, None # last item to compy with gym syntax

    
    def reset(self):
        self.time_step = 0

        if self.client:
            self.client.R.d['meta'] = 1
            self.client.respond_to_server()

            # TO avoid memory leak re-launch torcs from time to time
            if random.random() < 0.25:
               restart_torcs()
        
        self.client = snakeoil.Client(p=3001)
        self.client.MAX_STEPS = np.inf

        self.observation = self.client.get_servers_input()

        return self.observation


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



    def kill_torcs(self):
        os.system(f'pkill torcs')


    def parse_torcs_input(self, obs_dict: dict):
        keys = ['angle', 'focus', 'speedX', 'speedY', 'speedZ', 'track', 'trackPos']
        observation = collections.namedtuple('observation', keys)
        print(observation._fields)
        return observation(
            angle=np.array(obs_dict['angle'], dtype=np.float32)/PI,
            focus=np.array(obs_dict['focus'], dtype=np.float32)/200.,
            speedX=np.array(obs_dict['speedX'], dtype=np.float32)/self.max_speed,
            speedY=np.array(obs_dict['speedY'], dtype=np.float32)/self.max_speed,
            speedZ=np.array(obs_dict['speedZ'], dtype=np.float32)/self.max_speed,
            track=np.array(obs_dict['track'], dtype=np.float32)/200.,
            trackPos=np.array(obs_dict['trackPos'], dtype=np.float32)
        )