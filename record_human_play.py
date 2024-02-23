from torcs_python_client.torcs_env import TorcsEnv
import torcs_python_client.snakeoil3_gym as snakeoil
import numpy as np
import pygame
import pickle
import sys
import os

if __name__ == '__main__':
    snakeoil.set_textmode(False)
    snakeoil.set_tracks(["quickrace"])
    env = TorcsEnv()
    s = env.reset()
    obs_list, rew_list, act_list, done_list = [], [], [], []
    obs_list.append(s)
    pygame.init()
    display = pygame.display.set_mode((300, 300))
    prev_steer = 0
    prev_accel = 0
    while True:
        actions = np.array([0.,0.])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                with open(f'{os.getcwd()}/human_data/record.pickle', 'wb') as file:
                    pickle.dump([obs_list, rew_list, act_list, done_list], file, protocol=pickle.HIGHEST_PROTOCOL)
                pygame.quit()
                sys.exit()

        keys_pressed = pygame.key.get_pressed()

        if keys_pressed[pygame.K_UP]:
            prev_accel += 0.02
            actions[0] = min(prev_accel, 0.5) if prev_steer > 0.1 else prev_accel
        elif keys_pressed[pygame.K_DOWN]:
            prev_accel = -0.3 if prev_accel == 0 else prev_accel
            prev_accel -= 0.02
            actions[0] = prev_accel
        else:
            prev_accel = 0

        if keys_pressed[pygame.K_LEFT]:
            prev_steer = 0.075 if prev_steer == 0 else prev_steer
            prev_steer += 0.02
            actions[1] = prev_steer
        elif keys_pressed[pygame.K_RIGHT]:
            prev_steer = 0.075 if prev_steer == 0 else prev_steer
            prev_steer += 0.02
            actions[1] = -1*prev_steer
        else: 
            prev_steer = 0



        act_list.append(list(actions))
        obs, rew, done, _, _ = env.step(actions)
        obs_list.append(obs)
        rew_list.append(rew)
        done_list.append(done)
        if done:
            obs_list.append(env.reset())

        