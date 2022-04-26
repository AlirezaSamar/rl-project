import torch
from torch import Tensor,nn
import gym
from torch.distributions import Categorical
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
writer = SummaryWriter()
import numpy

from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris

NO_EPOCHS = 10000
EPOCH_STEPS = 2048
PPO_STEPS = 5
GAMMA = 0.99
LAMB = 0.95
CLIP = 0.2
lr = 3e-4
batch_size = 32

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)

class Critic(nn.Module):
    def __init__(self,obs, hidden_size = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
            )
        
        self.net.apply(init_weights)

    def forward(self,x):
        return self.net(x)

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
        
        self.net.apply(init_weights)

    def forward(self,x):
        logits = self.net(x)

        return logits

class ActorCritic(nn.Module):
    def __init__(self, critic, actor):
        super().__init__()

        self.critic = critic
        self.actor = actor

    def forward(self, state):
        logits = self.actor(state)
        val = self.critic(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        probs = dist.log_prob(action)
        return action, probs, val



env = gym.make("LunarLander-v2")
state = torch.Tensor(env.reset())
obs = env.observation_space.shape[0]
actions = env.action_space.n
actor = Actor(obs,actions)
critic = Critic(obs)



agent = ActorCritic(critic,actor)

opt = optim.Adam(agent.parameters(), lr=lr)

def gae(rewards, values):

    rs = rewards
    vals = values

    x = []
    for i in range(len(rs)-1):
        x.append(rs[i]+GAMMA*vals[i+1] - vals[i])

    a = discount(x, GAMMA * LAMB)
    return a

def discount(rewards, gamma):

    rs = []
    sum_rs = 0

    for r in reversed(rewards):
        sum_rs = (sum_rs * gamma) + r
        rs.append(sum_rs)


    return list(reversed(rs))

def update(states,actions,prob_old,vals,advs):

    _, prob, vals_new = agent(states)

    ratio = torch.exp(prob - prob_old)
    #PPO update
    clip = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advs
    #negative gradient descent - gradient ascent
    actor_loss = -(torch.min(ratio * advs, clip)).mean()
    #MSE
    clip2 = (torch.clamp(vals-vals_new,1 - CLIP, 1 + CLIP) - vals_new).pow(2)

    critic_loss = 0.5*torch.max(clip2, vals-vals_new).mean()

    opt.zero_grad()

    (critic_loss + actor_loss).backward()

    opt.step()

    return actor_loss, critic_loss



r_avg = deque(maxlen=100)
for e in range(NO_EPOCHS):
    states = []
    actions = []
    probs = []
    advs = []
    vals = []
    ep_rewards = []
    ep_vals = []
    epoch_rewards = []
    avg_reward = 0
    done = False
    state = torch.Tensor(env.reset())
    for i in range(EPOCH_STEPS):

        with torch.no_grad():
            action, ps, val = agent(state)

        next_state, reward, done, _ = env.step(action.item())

        states.append(state)
        actions.append(action)
        probs.append(ps)
        ep_rewards.append(reward)
        ep_vals.append(val)

        state = torch.Tensor(next_state)

        if (i == EPOCH_STEPS-1) or done:

            with torch.no_grad():
                _, _, val = agent(state)
                nxt = val

            ep_rewards.append(nxt)
            ep_vals.append(nxt)

            r_avg.append(sum(ep_rewards))

            vals += discount(ep_rewards,GAMMA)[:-1]
            advs += gae(ep_rewards,ep_vals)

            epoch_rewards.append(sum(ep_rewards))

            ep_rewards.clear()
            ep_vals.clear()

            state = torch.Tensor(env.reset())

    states = torch.stack((states))
    actions = torch.stack((actions))
    probs = torch.stack((probs))
    vals = torch.stack((vals))
    advs = torch.stack((advs))
    
    actor_loss, critic_loss = update(states,actions,probs,vals,advs)

    print("[ Epoch :",e,"- actor_loss: {:.2e}".format(actor_loss.item()),", critic_loss: {:.2e}".format(critic_loss.item()),", avg_reward: {:.2f} ".format((sum(epoch_rewards)/len(epoch_rewards)).item()), "running average: {:.2f}]    ".format(numpy.average(r_avg)), end='\r')

    writer.add_scalar("actor_loss",actor_loss,e)
    writer.add_scalar("critic_loss",critic_loss,e)
    writer.add_scalar("avg_reward",sum(epoch_rewards)/len(epoch_rewards),e)
    writer.flush()
    
    if numpy.average(r_avg) >=200 and len(r_avg) == 100:
        print("100 episode rolling average > 200, stopping...")
        exit()

writer.close()



