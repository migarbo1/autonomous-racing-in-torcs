from torch.distributions import MultivariateNormal
from .network import FeedForwardNN
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np
import torch


class PPO:
    def __init__(self, env) -> None:
        # Set hyperparameters
        self._init_hyperparameters()
        
        # Get enviroment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # create the covariance matrix for continuous action space
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # Define the optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)


    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005
        self.num_minibatches = 12
        self.ent_coef = 0


    def get_action(self, state):
        mean = self.actor(state)

        distribution = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

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
        future_rewards_batch = []   # shape(self.timesteps_per_batch,)
        ep_lengths_batch = []       # shape(num_episodes,)

        cur_timesteps_in_batch = 0
        while cur_timesteps_in_batch < self.timesteps_per_batch:
            ep_rewards = []

            #TODO: see if our env returns 1 or 2 args in reset
            state, _ = self.env.reset()
            done = False

            for step in range(self.max_timesteps_per_episode):
                cur_timesteps_in_batch += 1

                obs_batch.append(state)
                action, log_prob = self.get_action(state)
                #TODO: see if ours needs also 5 args
                state, reward, done, _ ,_ = self.env.step(action)

                act_batch.append(action)
                ep_rewards.append(reward)
                logprob_batch.append(log_prob)

                if done:
                    break

            ep_lengths_batch.append(cur_timesteps_in_batch + 1)
            rewards_batch.append(ep_rewards)

        obs_batch = torch.tensor(obs_batch, dtype=torch.float)
        act_batch = torch.tensor(act_batch, dtype=torch.float)
        logprob_batch = torch.tensor(logprob_batch, dtype=torch.float)

        future_rewards_batch = self.compute_future_rewards(rewards_batch)

        return obs_batch, act_batch, logprob_batch, future_rewards_batch, ep_lengths_batch


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


    def learn(self, max_steps):
        current_timesteps = 0
        while current_timesteps < max_steps:
            obs_batch, act_batch, logprob_batch, future_rewards_batch, ep_lengths_batch = self.rollout()

            V, _, _ = self.evaluate(obs_batch, act_batch)
            advantage_k = future_rewards_batch - V.detach()
            # Normatize advantage to make PPO stable
            advantage_k = (advantage_k - advantage_k.mean()) / (advantage_k.std() + 1e-10)

            step = obs_batch.size(0)
            indexes = np.arange(step)
            minibatch_size = step // self.num_minibatches

            for _ in range(self.n_updates_per_iteration):

                self.lr_annealing(current_timesteps, max_steps)

                np.random.shuffle(indexes)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = indexes[start:end]
                    mini_obs = obs_batch[idx]
                    mini_acts = act_batch[idx]
                    mini_logprobs = logprob_batch[idx]
                    mini_ak = advantage_k[idx]
                    mini_future_rewards = future_rewards_batch[idx]

                    # Compute pi_theta(a_t, s_t)
                    V, cur_logprob, entropy = self.evaluate(mini_obs, mini_acts)

                    # compute ratios
                    ratios = torch.exp(cur_logprob - mini_logprobs)


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
                self.actor_opt.step()

                #compute gradients and propagate loss though critic
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

            current_timesteps += np.sum(ep_lengths_batch)