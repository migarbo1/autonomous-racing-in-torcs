import os
import matplotlib.pyplot as plt
import pickle

def load_training_data():
    if os.path.isfile(f'{os.getcwd()}/training_data_focus_track.pickle'):
        with open(f'{os.getcwd()}/training_data_focus_track.pickle', 'rb') as file:
            return pickle.load(file)
    else:
        return {"total_training_timesteps": 0.0, "actor_episodic_avg_loss": [], "critic_episodic_avg_loss": [], "eval_results": []}

tr = load_training_data()

rewards = [a['total_reward'] for a in tr['eval_results']]

plt.plot(rewards)
plt.show()
