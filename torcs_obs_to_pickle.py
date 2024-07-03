import pickle
import numpy as np
import sys
import os

def compute_reward(speed_x, speed_y, angle, prev_speed, prev_angle):

    speed_dif = abs(speed_x - prev_speed)
    
    speed_reward = 2*speed_dif + speed_x * (np.cos(angle) - abs(np.sin(angle))) - abs(speed_y)*np.cos(angle)
    
    # get angle of first frame of stack
    angle_variation = 5*abs(angle - prev_angle)

    return speed_reward - angle_variation


if __name__ == '__main__':
    lines = []
    observation_list = []
    action_list = []
    done_list = []
    reward_list = []
    prev_angle = 0
    prev_speed = 0
    use_fs =  len(sys.argv) > 3 and sys.argv[3] == 'True'
    fs = [np.zeros(24,) for _ in range(5)]

    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()

    for line in lines:
        elements = line.split(',')
        angle = float(elements[0].replace('(angle ', '').replace(')', ''))
        speed_x = float(elements[1].replace('(speedX ', '').replace(')', ''))
        speed_y = float(elements[2].replace('(speedY ', '').replace(')', ''))
        speed_z = float(elements[3].replace('(speedZ ', '').replace(')', ''))
        track_pos = float(elements[4].replace('(trackPos ', '').replace(')', ''))
        track = elements[5].replace('(track ', '').replace(')', '').split(' ')
        for i in range(len(track)):
            track[i] = float(track[i])/200.

        steer = float(elements[6].replace('(Steer ', '').replace(')', ''))
        brake = float(elements[7].replace('(Brake ', '').replace(')', ''))
        accel = float(elements[8].replace('(Accel ', '').replace(')', ''))
        ab = accel if accel > 0 else -1.0*brake

        observation = [
            angle/np.pi,
            speed_x/300.,
            speed_y/300.,
            speed_z/300.,
            track_pos,
            *track
        ]

        if use_fs:
            aux = fs[1:]
            aux.append(observation)
            fs = aux.copy()

        if use_fs:
            observation_list.append(list(np.array(fs).flatten()))
        else:
            observation_list.append(observation)
        
        action = [
            ab,
            steer
        ]
        action_list.append(action)

        reward_list.append(compute_reward(speed_x/300., speed_y/300., angle, prev_speed, prev_angle))
        done_list.append(False)
        prev_angle = angle if not use_fs else fs[0][0]
        prev_speed = speed_x/300. if not use_fs else fs[0][1]

    res = [observation_list, reward_list, action_list, done_list]

    formatted_human_data = []
    formatted_human_data.append(res)

    with open(sys.argv[2], 'wb') as file:
        pickle.dump(formatted_human_data, file, protocol=pickle.HIGHEST_PROTOCOL)


