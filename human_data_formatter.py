import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np

matplotlib.style.use('bmh')

def load_training_data(name):
    if os.path.isfile(f'{os.getcwd()}/human_data/{name}.pickle'):
        with open(f'{os.getcwd()}/human_data/{name}.pickle', 'rb') as file:
            return pickle.load(file)

formatted_human_data = []

for n in ['record']:#['aalborg', 'suzuka', 'brondehach', 'corkscrew']:
    tr = load_training_data(n)

    obs = tr[0]
    rew = tr[1]
    action = tr[2]
    done = tr[3]

    print(f"obs: {len(obs)}")
    print(f"reward: {len(rew)}")
    print(f"action: {len(action)}")
    print(f"done: {len(done)}")

    end_idx = np.argmax(rew)

    obs = obs[:end_idx]
    rew = rew[:end_idx]
    action = action[:end_idx]
    done = done[:end_idx]

    print(f"obs: {len(obs)}")
    print(f"reward: {len(rew)}")
    print(f"action: {len(action)}")
    print(f"done: {len(done)}")

    start_idx = np.argmax(np.array(rew)>0)

    obs = obs[start_idx:]
    rew = rew[start_idx:]
    action = action[start_idx:]
    done = done[start_idx:]

    print(f"obs: {len(obs)}")
    print(f"reward: {len(rew)}")
    print(f"action: {len(action)}")
    print(f"done: {len(done)}")

    res = [obs, rew, action, done]
    formatted_human_data.append(res)

with open(f'{os.getcwd()}/human_data/formatted.pickle', 'wb') as file:
    pickle.dump(formatted_human_data, file, protocol=pickle.HIGHEST_PROTOCOL)
