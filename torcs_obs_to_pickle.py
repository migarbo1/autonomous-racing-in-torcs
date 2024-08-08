import numpy as np
import pickle
import click
import sys
import os

def compute_spbased_reward(speed_x, speed_y, angle, prev_speed, prev_angle, track_pos):

    if abs(track_pos) > 1.0:
        return -10000

    speed_dif = abs(speed_x - prev_speed)
    
    speed_reward = 2*speed_dif + speed_x * (np.cos(angle) - abs(np.sin(angle))) - abs(speed_y)*np.cos(angle)
    
    # get angle of first frame of stack
    angle_variation = 5*abs(angle - prev_angle)

    return speed_reward - angle_variation

def compute_distbased_reward(angle, track_pos, dist_raced, prev_dist_raced):

    if abs(track_pos) > 1.0:
        return -10000

    return (dist_raced-prev_dist_raced) * np.cos(angle) * (1 - abs(track_pos))

@click.command()
@click.option('--num_frames', '-f', default=1, help='Number of frames to be stacked in an observation')
@click.option('--reward_type', type=click.Choice(['speed', 'distance'], case_sensitive=False), default='speed')
@click.option('--join_accel_brake', '-j', is_flag=True, default=False, help='Combine accel and brake accions into a single one')
@click.option('--prefix', '-p', default='./human_data', help='folder containing the files')
def main(num_frames, reward_type, join_accel_brake, prefix):
    print(num_frames, reward_type, join_accel_brake, prefix)
    observation_list = []
    action_list = []
    done_list = []
    reward_list = []
    use_fs =  int(num_frames) > 1

    for obs_file in ['aalborg', 'bron', 'cork', 'wheel']:
        lines = []
        fs = [np.zeros(25,) for _ in range(num_frames)]

        with open(f'{prefix}_{obs_file}.txt', 'r') as file:
            lines = file.readlines()

        prev_dist_raced = 0
        prev_angle = 0
        prev_speed = 0

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
            dist_raced = float(elements[6].replace('(distRaced ', '').replace(')', ''))

            steer = float(elements[7].replace('(Steer ', '').replace(')', ''))
            brake = float(elements[8].replace('(Brake ', '').replace(')', ''))
            accel = float(elements[9].replace('(Accel ', '').replace(')', ''))
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
            
            if join_accel_brake: 
                action = [
                    ab,
                    steer
                ]
            else:
                action = [
                    accel,
                    brake,
                    steer
                ]
            action_list.append(action)

            if reward_type == 'speed':
                reward = compute_spbased_reward(speed_x/300., speed_y/300., angle, prev_speed, prev_angle, track_pos)
            else:
                reward = compute_distbased_reward(angle, track_pos, dist_raced, prev_dist_raced)
            reward_list.append(reward)
            
            prev_dist_raced = dist_raced
            done_list.append(abs(track_pos)>1.0)
            prev_angle = angle if not use_fs else fs[0][0]
            prev_speed = speed_x/300. if not use_fs else fs[0][1]

    assert len(observation_list) == len(reward_list) == len(action_list) == len(done_list)

    res = [observation_list, reward_list, action_list, done_list]

    formatted_human_data = []
    formatted_human_data.append(res)

    with open(f'{prefix}_{reward_type}_formatted.pickle', 'wb') as file:
        pickle.dump(formatted_human_data, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()