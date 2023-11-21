from torcs_env import TorcsEnv
import numpy as np
import random

episode_count = 10
max_steps = 50
reward = 0
done = False
step = 0

# Generate a Torcs environment
env = TorcsEnv()

def get_action_object():
    return {
        'accel': random.uniform(0, 1),
        'brake': 0,
        'steering': random.uniform(-1, 1)
    }

print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    state = env.reset()

    total_reward = 0.
    for j in range(max_steps):
        action = get_action_object()
       # print('action: ', action)
        ob, reward, done, _ = env.step(action)
        print('state:', ob)
        total_reward += reward

        step += 1
        if done:
            break

    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("")

env.kill_torcs()  # This is for shutting down TORCS
print("Finish.")
