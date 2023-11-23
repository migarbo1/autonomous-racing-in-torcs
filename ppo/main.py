import gymnasium as gym
from ppo import PPO

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', continuous = True)
    model = PPO(env)
    model.learn(100000)

# import gymnasium as gym
# env = gym.make("CartPole-v1")

# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()