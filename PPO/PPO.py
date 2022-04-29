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
PPO_STEPS = 10
GAMMA = 0.99
LAMB = 0.95
CLIP = 0.2
lr_a = 3e-4
lr_c = 1e-3

class Policy(nn.Module):

    def __init__(self, obs, n_actions, hidden_size = 128):

        super(Policy, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1),
        )


        self.critic = nn.Sequential(
            nn.Linear(obs,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self):
        pass

    def act(self,x):

        probs = self.actor(x)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach()

    def eval(self,states,actions):

        probs = self.actor(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        vals = self.critic(states)

        return log_probs, vals, entropy


env = gym.make("LunarLander-v2")##Tetris(grid_dims=(10,10), piece_size=2)#
obs = env.observation_space.shape[0]
actions = env.action_space.n

agent = Policy(obs,actions)

opt = optim.Adam([{'params': agent.actor.parameters(), 'lr': lr_a},
                {'params': agent.critic.parameters(), 'lr': lr_c}
                    ])

r_avg = deque(maxlen=100)
r_avg_len = deque(maxlen=100)

def update(states,actions,probs,rewards,dones):

    discount_r = []
    r = 0

    for rew, done in zip(reversed(rewards),reversed(dones)):
        if done:
            r = 0
        else:
            r = rew + (GAMMA*r)
        discount_r.insert(0,r)
    
    discount_r = torch.tensor(discount_r, dtype=torch.float32)
    #discount_r = (discount_r - discount_r.mean()) / (discount_r.std() + 1e-7)

    o_states = torch.squeeze(torch.stack(states,dim=0)).detach()
    o_actions = torch.squeeze(torch.stack(actions,dim=0)).detach()
    o_probs = torch.squeeze(torch.stack(probs,dim=0)).detach()

    losses = []

    for _ in range(PPO_STEPS):

        n_probs, vals, entropy = agent.eval(o_states,o_actions)

        vals = torch.squeeze(vals)

        ratio = torch.exp(n_probs - o_probs.detach())

        adv = discount_r - vals.detach()

        clip = torch.clamp(ratio, 1-CLIP, 1+CLIP) * adv

        loss = -torch.min(ratio*adv, clip) + 0.5*nn.MSELoss()(vals,discount_r) - 0.01*entropy

        opt.zero_grad()
        loss.mean().backward()
        opt.step()

        losses.append(loss.mean())
    
    return sum(losses)/len(losses)

    

for e in range(NO_EPOCHS):

    states = []
    actions = []
    probs = []
    rewards = []
    dones = []
    ep_reward = 0
    ep_len = 0
    state = torch.Tensor(env.reset())

    for i in range(EPOCH_STEPS):

        with torch.no_grad():
            action, log_prob = agent.act(state)

        next_state, reward, done, _ = env.step(action.item())

        ep_reward += reward
        ep_len += 1

        states.append(state)
        actions.append(action)
        probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)

        state = torch.Tensor(next_state)

        if done:
            state = torch.Tensor(env.reset())
            r_avg.append(ep_reward)
            r_avg_len.append(ep_len)
            ep_len = 0
            ep_reward = 0

    loss = update(states,actions,probs,rewards,dones)  

    print("[ Epoch :",e,"- loss: {:.2e}".format(loss.item()),", running average: {:.2f}]    ".format((sum(r_avg)/len(r_avg))),", running average len: {:.2f}]    ".format((sum(r_avg_len)/len(r_avg))), end='\r')








