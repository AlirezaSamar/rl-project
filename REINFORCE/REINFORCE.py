import torch
from torch import Tensor,nn
import gym
from torch.distributions import Categorical
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
writer = SummaryWriter()
import numpy

NO_EPOCHS = 10000

GAMMA = 0.99
LAMB = 0.95
CLIP = 0.2
lr = 1e-2

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
    cum_r = 0
    ret = []
    loss = []

    for r in reversed(rewards):
        cum_r = r + GAMMA*cum_r
        ret.append(cum_r)
    ret = torch.tensor(ret[::-1])
    ret = (ret - ret.mean()) / (ret.std() + 1e-8)
    for p, r in zip(probs,ret):
        loss.append(-p*r)
    opt.zero_grad()
    loss = torch.cat(loss).sum()
    loss.backward()
    opt.step()

    return loss

tot_rewards = deque(maxlen=100)
for i in range(NO_EPOCHS):
    state = env.reset()
    ep_reward = 0
    probs = []
    rewards = []
    done = False
    while not done:
        dist, action = actor(torch.Tensor(state).unsqueeze(0))

        prob = dist.log_prob(action)
        probs.append(prob[None,...])

        state, reward, done, _ = env.step(action.item())

        rewards.append(reward)

        ep_reward += reward

    loss = update(probs,rewards)
    tot_rewards.append(ep_reward)

    if i%10 == 0:
        print("Episode {}".format(i),", loss : {:.2f}".format(loss),", ep_reward: {:.2f}".format(ep_reward),", average : {:.2f}".format(numpy.average(tot_rewards)))

    if numpy.average(tot_rewards) >=200 and len(tot_rewards) == 100:
        print("100 episode rolling average > 200, stopping...")
        exit()

