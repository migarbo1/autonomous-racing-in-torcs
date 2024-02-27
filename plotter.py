import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np

matplotlib.style.use('bmh')

def load_training_data(name):
    if os.path.isfile(f'{os.getcwd()}/{name}.pickle'):
        with open(f'{os.getcwd()}/{name}.pickle', 'rb') as file:
            return pickle.load(file)
    else:
        return {"total_training_timesteps": 0.0, "actor_episodic_avg_loss": [], "critic_episodic_avg_loss": [], "eval_results": []}

# plot comparativo rewards por episodio
# tr1 = load_training_data('training_data_baseline')
# tr3 = load_training_data('training_data_fs_3')
# tr5 = load_training_data('training_data_fs_5')

# rewards_tr1 = [a['rewards_per_timestep'] for a in tr1['eval_results']]
# rewards_tr3 = [a['rewards_per_timestep'] for a in tr3['eval_results']]
# rewards_tr5 = [a['rewards_per_timestep'] for a in tr5['eval_results']]

# rewards_tr1 = [sum(a[:]) for a in rewards_tr1]
# rewards_tr3 = [sum(a[:]) for a in rewards_tr3]
# rewards_tr5 = [sum(a[:]) for a in rewards_tr5]

# plt.title('Reward earned in eval track')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.plot([a for a in range(len(rewards_tr1))],rewards_tr1, '--', label='baseline')
# plt.plot([a for a in range(len(rewards_tr5)-2)],rewards_tr5[:-2], '-',label='5 frames')
# plt.plot([a for a in range(len(rewards_tr3))],rewards_tr3, '-.',label='3 frames')
# plt.legend()
# plt.show()

# plot comparativo rollout por episodio
# tr1 = load_training_data('training_data_baseline')
# tr3 = load_training_data('training_data_fs_3')
# tr5 = load_training_data('training_data_fs_5')

# rewards_tr1 = [a['dist_raced'] for a in tr1['eval_results']]
# rewards_tr3 = [a['dist_raced'] for a in tr3['eval_results']]
# rewards_tr5 = [a['dist_raced'] for a in tr5['eval_results']]

# plt.title('Distance covered in eval track')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.plot([a for a in range(len(rewards_tr1)-4)],rewards_tr1[:-4], '--', label='baseline')
# plt.plot([a for a in range(len(rewards_tr5)-2)],rewards_tr5[:-2], '-',label='5 frames')
# plt.plot([a for a in range(len(rewards_tr3))],rewards_tr3, '-.',label='3 frames')
# plt.legend()
# plt.show()

#plot comparativo loss actor entre modelos fs

# tr1 = load_training_data('training_data_baseline')
# tr3 = load_training_data('training_data_fs_3')
# tr5 = load_training_data('training_data_fs_5')
# plt.title('Actor episodic average loss')
# plt.xlabel('Episode')
# plt.ylabel('Actor loss')
# plt.plot(tr1['actor_episodic_avg_loss'], label='baseline')
# plt.plot(tr3['actor_episodic_avg_loss'], label='fs_3')
# plt.plot(tr5['actor_episodic_avg_loss'], label='fs_5')
# plt.legend()
# plt.show()
