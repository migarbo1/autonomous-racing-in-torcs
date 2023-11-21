from torcs_env import TorcsEnv
import numpy as np

episode_count = 10
max_steps = 50
reward = 0
done = False
step = 0

# Generate a Torcs environment
env = TorcsEnv()

agent = Agent(1)  # steering only


print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    if np.mod(i, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    total_reward = 0.
    for j in range(max_steps):
        action = agent.act(ob, reward, done, vision)
       # print('action: ', action)
        ob, reward, done, _ = env.step(action)
       # print('state:', ob)
        total_reward += reward

        step += 1
        if done:
            break

    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("")

env.end()  # This is for shutting down TORCS
print("Finish.")
