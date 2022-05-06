import torch
from torch import Tensor,nn
import gym
from torch.distributions import Categorical
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
writer = SummaryWriter()
import numpy

NO_EPOCHS = 5000
EP_STEPS = 1000

GAMMA = 0.99
lr = 1e-3

class Actor(nn.Module):
    def __init__(self,obs, n_actions, hidden_size = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
            )

    def forward(self,x):
        logits = self.net(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return dist,action


env = gym.make("LunarLander-v2")
obs = env.observation_space.shape[0]
actions = env.action_space.n
actor = Actor(obs,actions)
opt = optim.Adam(actor.parameters(), lr=lr)

def update(probs, rewards):
    ret = []

    for t in range(len(rewards)):
        cum_r = 0
        pw = 0
        for r in rewards[t:]:
            cum_r = cum_r + GAMMA**pw*r
            pw += 1
        ret.append(cum_r)

    ret = torch.tensor(ret)
    ret = (ret - ret.mean()) / (ret.std())

    probs = torch.stack(probs)
    loss = -probs*ret

    opt.zero_grad()
    loss = loss.sum()
    loss.backward()
    opt.step()

    return loss

def evaluate():
    test_env = gym.make("LunarLander-v2")

    for _ in range(20):
        state = torch.Tensor(env.reset())
        done = False
        while not done:
            with torch.no_grad():
                _, action = actor(state)
                next_state, reward, done, _ = env.step(action.item())
                state = torch.Tensor(next_state)
                env.render()

tot_rewards = deque(maxlen=100)
avg_len =  deque(maxlen=100)
ep_count = 0
for i in range(NO_EPOCHS):
    state = env.reset()
    ep_reward = 0
    ep_len = 0
    probs = []
    rewards = []
    done = False
    if ep_count == 2000:
        evaluate()
        exit()
    for _ in range(EP_STEPS):
        dist, action = actor(torch.Tensor(state))

        prob = dist.log_prob(action)
        probs.append(prob)

        state, reward, done, _ = env.step(action.item())
        rewards.append(reward)

        ep_reward += reward
        ep_len += 1

        if done:
            break

    ep_count += 1

    loss = update(probs,rewards)
    tot_rewards.append(ep_reward)
    avg_len.append(ep_len)
    ep_len = 0

    writer.add_scalar("avg_reward",numpy.average(tot_rewards),ep_count)
    writer.add_scalar("loss",loss,ep_count)
    writer.add_scalar("avg_len",numpy.average(avg_len),ep_count)

    if i%10 == 0:
        print("Episode {}".format(i),", loss : {:.2f}".format(loss),", ep_reward: {:.2f}".format(ep_reward),", average : {:.2f}".format(numpy.average(tot_rewards)))

    if (numpy.average(tot_rewards) >=200 and len(tot_rewards) == 100) or ep_count >= 5000:
        print("100 episode rolling average > 200, stopping...")
        writer.close()
        exit()

writer.close()
