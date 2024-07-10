import time
from torch.distributions import MultivariateNormal
from .network import Actor, Critic
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np
import random
import pickle
import torch
import os


class PPO:
    def __init__(self, env, variance=0.35, test = False, use_human_data = False, default_lr = 0.005, model_name='ppo', try_brake = False, human_data_file='', human_steps = 0) -> None:
        # Set hyperparameters
        self._init_hyperparameters(default_lr)
        
        # Get enviroment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.use_human_data = use_human_data
        self.try_brake = try_brake

        # Initialize actor and critic networks
        self.actor = Actor(self.obs_dim, self.act_dim, test=test)
        self.critic = Critic(self.obs_dim, 1, test=test)

        self.model_name = model_name

        if os.path.isfile(f'{model_name}_actor.pth'):
            print('loading actor weights ....')
            self.actor.load_state_dict(torch.load(f'{model_name}_actor.pth'))

        if os.path.isfile(f'{model_name}_critic.pth'):
            print('loading critic weights ....')
            self.critic.load_state_dict(torch.load(f'{model_name}_critic.pth'))

        if self.use_human_data and os.path.isfile(human_data_file):
            print('loading human data...')
            with open(human_data_file, 'rb') as file:
                self.human_data = pickle.load(file)
                
        self.actor.to('cuda')
        self.critic.to('cuda')

        self.min_variance = 0.01

        self.variance = variance if not test else self.min_variance
        self.curr_variance = variance if not test else self.min_variance

        # Define the optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)

        self.test = test
        self.current_timesteps = 0
        self.human_steps = human_steps


    def _init_hyperparameters(self, lr):
        self.timesteps_per_batch = 19200 # 48
        self.max_timesteps_per_episode = 6400 # 16
        self.gamma = 0.99
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = lr
        self.save_ratio = 3

        self.eval_ratio = 3
        self.eval_max_timesteps = 15000

        # advanced hyper parameters 
        self.num_minibatches = 12
        self.ent_coef = 0.05
        self.max_grad_norm = 0.5
        self.target_kl = 0.02 
        self.lamda = 0.98


    def get_action(self, state):
        mean = self.actor(state)
        
        # create the covariance matrix for continuous multi action space
        cov_var = torch.full(size=(self.act_dim,), fill_value=self.curr_variance)
        cov_mat = torch.diag(cov_var)

        distribution = MultivariateNormal(mean, cov_mat)

        # Sample an action from the distribution and get its log prob
        action = distribution.sample()

        # stochastic breaking
        if not self.test and self.try_brake:
            rate = (self.current_timesteps*2)/self.max_timesteps if self.current_timesteps < self.max_timesteps/2 else 1-self.current_timesteps/self.max_timesteps
            if random.random() < 0.1 * rate:
                if self.act_dim ==2:
                    action[0] = min(rate*-0.1, action[0])
                else:
                    action[1] = min(rate*0.1, action[1])

        # prevent the agent from braking if it's driving at or slower than the min
        if not self.test and state[-23] < 3/self.env.max_speed: 
            if self.act_dim ==2:
                action[0] = max(0, action[0])
            else:
                action[1] = 0

        if self.test:
            return action.detach().cpu().numpy(), 1

        log_prob = distribution.log_prob(action)
        #detach because they are tensors
        return action.detach().cpu().numpy(), log_prob.detach().cpu()


    def compute_future_rewards(self, rewards_batch):
        future_rewards_batch = []

        for episode_rewards in reversed(rewards_batch):
            discounted_episode_reward = 0

            for reward in reversed(episode_rewards):
                discounted_episode_reward = reward + discounted_episode_reward * self.gamma
                future_rewards_batch.insert(0, discounted_episode_reward)
        
        future_rewards_batch = torch.tensor(future_rewards_batch, dtype=torch.float)

        return future_rewards_batch


    def launch_eval(self, only_practice=True):
        eval_results = {}

        i = 0
        max_speed = 0
        speed_list = []
        obs_list = []
        total_reward = 0
        reward_list = []
        laps_completed = 0
        last_lap_time = 0 

        done = False
        self.test = True
        state, _ = self.env.reset(eval=only_practice)
        while not done and i < self.eval_max_timesteps:
            action, log_prob = self.get_action(state)
            state, reward, done, _ ,_ = self.env.step(action)

            total_reward += reward
            reward_list.append(reward)
            obs_list.append(self.env.previous_state) #store previous_state so we have all the information. At this time prev_state = obs_after_action

            if reward < -10 and done:
                print('\nepisode terminated by a collision')

            curr_speed = self.env.previous_state['speedX']
            speed_list.append(curr_speed)
            if curr_speed > max_speed:
                max_speed = curr_speed

            state_last_lap_time = self.env.previous_state['lastLapTime']
            if state_last_lap_time != last_lap_time:
                last_lap_time = state_last_lap_time
                laps_completed += 1

            i+=1
        if i==self.eval_max_timesteps:
            print('max eval timesteps  reached, stopping rollout...')

        eval_results['max_speed'] = max_speed
        eval_results['avg_speed'] = np.mean(speed_list)
        eval_results['dist_raced'] = obs_list[-1]['distRaced']
        eval_results['laps_completed'] = laps_completed
        eval_results['rewards_per_timestep'] = reward_list
        eval_results['total_reward'] = total_reward
        eval_results['observation_rollout'] = obs_list
        eval_results['all_steps_completed'] = i==self.eval_max_timesteps

        # self.env.training_data['eval_results'].append(eval_results)
        self.test = False
        return eval_results


    def rollout(self):
        obs_batch = []              # shape(self.timesteps_per_batch, self.obs_dim)
        act_batch = []              # shape(self.timesteps_per_batch, self.act_dim)
        logprob_batch = []          # shape(self.timesteps_per_batch,)
        rewards_batch = []          # shape(num_episodes, self.max_timesteps_per_episode)
        ep_lengths_batch = []       # shape(num_episodes,)

        val_batch = []
        dones_batch = []
        cur_timesteps_in_batch = 0

        while cur_timesteps_in_batch < self.timesteps_per_batch:
            ep_rewards = []
            ep_vals = []
            ep_dones = []

            state, _ = self.env.reset()
            done = False

            for k in range(self.max_timesteps_per_episode):

                ep_dones.append(done)

                cur_timesteps_in_batch += 1

                obs_batch.append(state)
                action, log_prob = self.get_action(state)
                
                val = self.critic(state)

                state, reward, done, _ ,_ = self.env.step(action)

                ep_vals.append(val.flatten())
                act_batch.append(action)
                ep_rewards.append(reward)
                logprob_batch.append(log_prob)

                if done:
                    break

            ep_lengths_batch.append(k + 1)
            rewards_batch.append(ep_rewards)
            val_batch.append(ep_vals)
            dones_batch.append(ep_dones)

        self.env.kill_torcs()

        obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float)
        act_batch = torch.tensor(np.array(act_batch), dtype=torch.float)
        logprob_batch = torch.tensor(np.array(logprob_batch), dtype=torch.float).flatten()

        return obs_batch, act_batch, logprob_batch, rewards_batch, ep_lengths_batch, val_batch, dones_batch


    def evaluate(self, obs_batch, act_batch):

        V = self.critic(obs_batch)

        # get logprobs using last actor network
        mean = self.actor(obs_batch)

        cov_var = torch.full(size=(self.act_dim,), fill_value=self.curr_variance)
        cov_mat = torch.diag(cov_var)

        distribution = MultivariateNormal(mean, cov_mat)

        logprobs = distribution.log_prob(act_batch)

        # Squeeze because we have shape(timesteps_per_batch, 1) and we hant only a 1d array 
        return V.squeeze(), logprobs, distribution.entropy()


    def lr_annealing(self, cur_timesteps, max_timesteps):
        frac = (cur_timesteps - 1.0) / max_timesteps
        new_lr = self.lr * (1.0 - frac)
        new_lr = max(new_lr, 0.0)
        self.actor_opt.param_groups[0]["lr"] = new_lr
        self.critic_opt.param_groups[0]["lr"] = new_lr

    def variance_decay(self, cur_timesteps, max_timesteps):
        frac = (cur_timesteps - 1.0) / max_timesteps
        self.curr_variance = self.variance * (1 - frac)
        self.curr_variance = max(self.curr_variance, self.min_variance)


    def compute_gae(self, rewards_in_batch, values_in_batch, dones_in_batch):
        advantage_batch = []

        for ep_r, ep_v, ep_d in zip(rewards_in_batch, values_in_batch, dones_in_batch):
            ep_advantages = []
            _advantage = 0

            for i in reversed(range(len(ep_r))):
                if i+1 < len(ep_r):
                    # TD error
                    delta = ep_r[i] + self.gamma * ep_v[i+1] * (1-ep_d[i+1]) - ep_v[i]
                else:
                    # last step
                    delta = ep_r[i] - ep_v[i]

                advantage = delta + self.gamma * self.lamda * (1-ep_d[i])* _advantage
                _advantage = advantage
                ep_advantages.insert(0, advantage)

            advantage_batch.extend(ep_advantages)
        
        return torch.tensor(advantage_batch, dtype=torch.float)


    def add_human_data(self, obs_batch, act_batch, logprob_batch, rewards_batch, val_batch, dones_batch):
        selected_human_data = random.sample(self.human_data, 1)[0]

        hd_obs = np.array(selected_human_data[0])
        hd_reward = selected_human_data[1]
        hd_actions = selected_human_data[2]
        hd_done = selected_human_data[3]

        idx = random.randint(0, len(hd_obs) - self.human_steps) - 1
        
        complete_obs_batch = torch.cat((obs_batch, torch.tensor(hd_obs[idx:idx+self.human_steps], dtype=torch.float)))
        complete_act_batch = torch.cat((act_batch, torch.tensor(hd_actions[idx:idx+self.human_steps], dtype=torch.float)))
        
        #tecnicament es com si sols tingueres un episodi i el ficares en la llista de episodis
        complete_done_batch = dones_batch + [hd_done[idx:idx+self.human_steps]]
        complete_rew_batch =  rewards_batch + [hd_reward[idx:idx+self.human_steps]]

        hd_logprobs = []
        hd_vals = []

        cov_var = torch.full(size=(self.act_dim,), fill_value=self.curr_variance)
        cov_mat = torch.diag(cov_var)
        
        for i in range(idx, idx+self.human_steps):
            action = torch.tensor(hd_actions[i], dtype=torch.float)
            distribution = MultivariateNormal(action, cov_mat)
            hd_logprobs.append(distribution.log_prob(action).detach().cpu())
            
            state = hd_obs[i]
            hd_vals.append(self.critic(state).flatten())

        complete_logprob_batch = torch.cat((logprob_batch, torch.tensor(hd_logprobs, dtype=torch.float).flatten()))
        complete_val_batch = val_batch + [hd_vals]

        return complete_obs_batch, complete_act_batch, complete_logprob_batch, complete_rew_batch, complete_val_batch, complete_done_batch
        

    def learn(self, max_timesteps):
        self.current_timesteps = 0
        current_iterations = 0
        timesteps_since_save = 0
        best_reward = -np.inf
        self.max_timesteps = max_timesteps
        while self.current_timesteps < max_timesteps:
            print(f'Current timesteps: {self.current_timesteps} out of {max_timesteps} | {self.current_timesteps/max_timesteps*100:.2f}%')
            obs_batch, act_batch, logprob_batch, rewards_batch, ep_lengths_batch, val_batch, dones_batch = self.rollout()

            if self.use_human_data and random.random() > 0.5*(self.current_timesteps/max_timesteps):
                obs_batch, act_batch, logprob_batch, rewards_batch, val_batch, dones_batch = self.add_human_data(obs_batch, act_batch, logprob_batch, rewards_batch, val_batch, dones_batch)

            # Compute Advantage using GAE
            advantage_k = self.compute_gae(rewards_batch, val_batch, dones_batch)
            V = self.critic(obs_batch).squeeze()
            future_rewards_batch = advantage_k + V.detach()

            # Normatize advantage to make PPO stable
            advantage_k = (advantage_k - advantage_k.mean()) / (advantage_k.std() + 1e-10)
            
            timesteps_in_batch = np.sum(ep_lengths_batch)
            self.current_timesteps += timesteps_in_batch
            timesteps_since_save += timesteps_in_batch

            current_iterations += 1

            step = obs_batch.size(0)
            indexes = np.arange(step)
            minibatch_size = step // self.num_minibatches

            iteration_actor_loss = []
            iteration_critic_loss = []

            self.lr_annealing(self.current_timesteps, max_timesteps)
            self.variance_decay(self.current_timesteps, max_timesteps)

            for _ in range(self.n_updates_per_iteration):

                np.random.shuffle(indexes)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = indexes[start:end]

                    # Get data from indexes
                    mini_obs = obs_batch[idx]
                    mini_acts = act_batch[idx]
                    mini_logprobs = logprob_batch[idx]
                    mini_ak = advantage_k[idx]
                    mini_future_rewards = future_rewards_batch[idx]

                    # Compute pi_theta(a_t, s_t)
                    V, cur_logprob, entropy = self.evaluate(mini_obs, mini_acts)

                    # compute ratios
                    log_ratios = cur_logprob - mini_logprobs
                    ratios = torch.exp(log_ratios)

                    approx_kl = ((ratios - 1) - log_ratios).mean()

                    # Compute surrogate losses
                    surr_1 = ratios * mini_ak
                    # Clamp == clip
                    surr_2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * mini_ak

                    # compute losses for both networks
                    actor_loss = (-torch.min(surr_1, surr_2)).mean()
                    # entropy regularization for actor network
                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_loss
                    
                    critic_loss = MSELoss()(V, mini_future_rewards)
                
                    # save actor and critic loss for logginfg purposes
                    iteration_actor_loss.append(actor_loss.item())
                    iteration_critic_loss.append(critic_loss.item())

                    # Propagate loss through actor network
                    self.actor_opt.zero_grad()
                    actor_loss.backward(retain_graph=True) #flag needed -> avoids buffer already free error
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_opt.step()

                    #compute gradients and propagate loss though critic
                    self.critic_opt.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    
                    self.critic_opt.step()

                # if approx_kl > self.target_kl:
                #     print('break')
                #     break

            # iteration_avg_actor_loss = np.mean(np.array(iteration_actor_loss))
            # iteration_avg_critic_loss = np.mean(np.array(iteration_critic_loss))

            # self.env.training_data['actor_episodic_avg_loss'].append(iteration_avg_actor_loss)
            # self.env.training_data['critic_episodic_avg_loss'].append(iteration_avg_critic_loss)

            print('current iteration: ', current_iterations)

            if current_iterations % self.eval_ratio == 0:
                res = self.launch_eval()
                rollout_reward = sum(res['rewards_per_timestep'][:-1])
                if rollout_reward > best_reward:
                    best_reward = rollout_reward
                    # self.env.training_data['total_training_timesteps'] += timesteps_since_save
                    # self.env.save_training_data()
                    timesteps_since_save = 0
                    torch.save(self.actor.state_dict(), f'{self.model_name}_actor.pth')
                    torch.save(self.critic.state_dict(), f'{self.model_name}_critic.pth')
                    print('Models saved')
        
        torch.save(self.actor.state_dict(), f'{self.model_name}_last_actor.pth')
        torch.save(self.critic.state_dict(), f'{self.model_name}_last_critic.pth')
