import torch
from torch import Tensor,nn
import gym
from torch.distributions import Categorical
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy

NO_EPOCHS = 10000
NO_STEPS = 2048
GAMMA = 0.99
LAMB = 0.95
CLIP = 0.2
lr = 3e-4
batch_size = 64

class Critic(nn.Module):
    def __init__(self,obs, hidden_size = 100):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
            )

    def forward(self,x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self,obs, n_actions, hidden_size = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
            )
    def forward(self,x):
        logits = self.net(x)
        logits = torch.nan_to_num(logits)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return dist,action

class ActorCritic():
    def __init__(self, critic, actor):
        self.critic = critic
        self.actor = actor

    @torch.no_grad()
    def __call__(self, state):
        dist, action = self.actor(state)
        probs = dist.log_prob(action)
        val = self.critic(state)

        return dist, action, probs, val



env = gym.make("LunarLander-v2")
state = torch.Tensor(env.reset())
obs = env.observation_space.shape[0]
actions = env.action_space.n

actor = Actor(obs,actions)
critic = Critic(obs)

actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

agent = ActorCritic(critic,actor)

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

    advs = (advs - advs.mean())/advs.std()

    dist, _ = actor(states)
    prob = dist.log_prob(actions)
    ratio = torch.exp(prob - prob_old)
    #PPO update
    clip = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advs
    #negative gradient descent - gradient ascent
    actor_loss = -(torch.min(ratio * advs, clip)).mean()

    vals_new = critic(states)
    #MSE
    critic_loss = (vals - vals_new).pow(2).mean()

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    actor_loss.backward()
    critic_loss.backward()

    actor_optimizer.step()
    critic_optimizer.step()

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
    for i in range(NO_STEPS):
        _, action, ps, val = agent(state)
        next_state, reward, done, _ = env.step(action.item())

        states.append(state)
        actions.append(action)
        probs.append(ps)
        ep_rewards.append(reward)
        ep_vals.append(val.item())

        state = torch.Tensor(next_state)

        if done or i==NO_STEPS-1:

            if i==NO_STEPS-1 and not done:

                #bootstrap value of last state if epoch ends early
                with torch.no_grad():
                    _,_,_,val = agent(state)
                    new_val = val.item()
            else:
                new_val = 0
                
            if done: 
                r_avg.append(sum(ep_rewards))

            #reward is approximated by value function if bootstrap, otherwise no reward for end of episode
            ep_rewards.append(new_val)
            ep_vals.append(new_val)

            vals += discount(ep_rewards,GAMMA)[:-1]
            advs += gae(ep_rewards,ep_vals)

            epoch_rewards.append(sum(ep_rewards))

            ep_rewards.clear()
            ep_vals.clear()
            state = torch.Tensor(env.reset())

    states = torch.stack((states))
    actions = torch.stack((actions))
    probs = torch.stack((probs))
    vals = torch.Tensor(vals)
    advs = torch.Tensor(advs)

    for i in range(0,NO_STEPS-batch_size,batch_size):
        actor_loss, critic_loss = update(states[i:i+batch_size],
                                        actions[i:i+batch_size],
                                        probs[i:i+batch_size],
                                        vals[i:i+batch_size],
                                        advs[i:i+batch_size])
    print("[ Epoch :",e,"- actor_loss: {:.2e}".format(actor_loss.item()),", critic_loss: {:.2e}".format(critic_loss.item()),", avg_reward: {:.2f} ]".format(sum(epoch_rewards)/len(epoch_rewards)), end='\r')

    writer.add_scalar("actor_loss",actor_loss,e)
    writer.add_scalar("critic_loss",critic_loss,e)
    writer.add_scalar("avg_reward",sum(epoch_rewards)/len(epoch_rewards),e)
    writer.flush()
    
    if numpy.average(r_avg) >=200 && len(r_avg) == 100:
        print("100 episode rolling average > 200, stopping...")
        exit()

writer.close()



