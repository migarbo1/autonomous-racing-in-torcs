import time
from torch.distributions import MultivariateNormal
from .network import FeedForwardNN
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np
import torch
import os


class PPO:
    def __init__(self, env, test = False) -> None:
        # Set hyperparameters
        self._init_hyperparameters()
        
        # Get enviroment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        if os.path.isfile('./ppo_actor.pth'):
            print('loading actor weights ....')
            self.actor.load_state_dict(torch.load('./ppo_actor.pth'))
        if os.path.isfile('./ppo_critic.pth'):
            print('loading critic weights ....')
            self.critic.load_state_dict(torch.load('./ppo_critic.pth'))

        # create the covariance matrix for continuous action space
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # Define the optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)

        self.test = test


    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800 # 120
        self.max_timesteps_per_episode = 1600 # 48
        self.gamma = 0.99
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005
        self.save_ratio = 3

        # advanced hyper parameters 
        self.num_minibatches = 6
        self.ent_coef = 0.05
        self.max_grad_norm = 0.5
        self.target_kl = 0.02 
        self.lamda = 0.98


    def get_action(self, state):
        mean = self.actor(state)

        # print('Action means: ', mean)

        distribution = MultivariateNormal(mean, self.cov_mat)   

        # Sample an action from the distribution and get its log prob
        action = distribution.sample()

        print('Selected actions: ', action)
        log_prob = distribution.log_prob(action)

        if self.test:
            return action.detach().numpy(), 1

        #detach because they are tensors
        return action.detach().numpy(), log_prob.detach()


    def compute_future_rewards(self, rewards_batch):
        future_rewards_batch = []

        for episode_rewards in reversed(rewards_batch):
            discounted_episode_reward = 0

            for reward in reversed(episode_rewards):
                discounted_episode_reward = reward + discounted_episode_reward * self.gamma
                future_rewards_batch.insert(0, discounted_episode_reward)
        
        future_rewards_batch = torch.tensor(future_rewards_batch, dtype=torch.float)

        return future_rewards_batch


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
            
            for _ in range(self.max_timesteps_per_episode):

                ep_dones.append(done)

                cur_timesteps_in_batch += 1

                obs_batch.append(state)
                action, log_prob = self.get_action(state)
                
                val = self.critic(state)

                #TODO: see if ours needs also 5 args
                state, reward, done, _ ,_ = self.env.step(action)

                ep_vals.append(val.flatten())
                act_batch.append(action)
                ep_rewards.append(reward)
                logprob_batch.append(log_prob)

                if done:
                    break

            ep_lengths_batch.append(cur_timesteps_in_batch + 1)
            rewards_batch.append(ep_rewards)
            val_batch.append(ep_vals)
            dones_batch.append(ep_dones)

        obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float)
        act_batch = torch.tensor(np.array(act_batch), dtype=torch.float)
        logprob_batch = torch.tensor(np.array(logprob_batch), dtype=torch.float).flatten()

        return obs_batch, act_batch, logprob_batch, rewards_batch, ep_lengths_batch, val_batch, dones_batch


    def evaluate(self, obs_batch, act_batch):

        V = self.critic(obs_batch)

        # get logprobs using last actor network
        mean = self.actor(obs_batch)
        distribution = MultivariateNormal(mean, self.cov_mat)
        logprobs = distribution.log_prob(act_batch)

        # Squeeze because we have shape(timesteps_per_batch, 1) and we hant only a 1d array 
        return V.squeeze(), logprobs, distribution.entropy()


    def lr_annealing(self, cur_timesteps, max_timesteps):
        frac = (cur_timesteps - 1.0) / max_timesteps
        new_lr = self.lr * (1.0 - frac)
        new_lr = max(new_lr, 0.0)
        self.actor_opt.param_groups[0]["lr"] = new_lr
        self.critic_opt.param_groups[0]["lr"] = new_lr


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


    def learn(self, max_timesteps):
        current_timesteps = 0
        current_iterations = 0
        while current_timesteps < max_timesteps:
            obs_batch, act_batch, logprob_batch, rewards_batch, ep_lengths_batch, val_batch, dones_batch = self.rollout()

            # Compute Advantage using GAE
            advantage_k = self.compute_gae(rewards_batch, val_batch, dones_batch)
            V = self.critic(obs_batch).squeeze()
            future_rewards_batch = advantage_k + V.detach()

            # Normatize advantage to make PPO stable
            advantage_k = (advantage_k - advantage_k.mean()) / (advantage_k.std() + 1e-10)
            
            current_timesteps += np.sum(ep_lengths_batch)
            current_iterations += 1

            step = obs_batch.size(0)
            indexes = np.arange(step)
            minibatch_size = step // self.num_minibatches

            for _ in range(self.n_updates_per_iteration):

                self.lr_annealing(current_timesteps, max_timesteps)

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

                if approx_kl > self.target_kl:
                    break
            print('current iterations:', current_iterations)
            if current_iterations % self.save_ratio == 0:
                print('Models saved')
                self.env.save_laptimes()
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')
