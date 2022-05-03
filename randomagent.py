import gym
import random

env = gym.make("LunarLander-v2")

ep_rewards = []
ep_lens = []
for _ in range(5000):
    done = False
    ep_reward = 0
    ep_len = 0
    state = env.reset()
    while not done:

        state, reward, done, _ = env.step(random.randint(0,env.action_space.n-1))

        ep_reward += reward
        ep_len += 1

    ep_rewards.append(ep_reward)
    ep_lens.append(ep_len)

with open("/home/scrungus/Documents/code/rl-project/results.txt", 'w+') as f:
    for r,l in zip(ep_rewards,ep_lens):
        f.write(str(r)+','+str(l)+'\n')


