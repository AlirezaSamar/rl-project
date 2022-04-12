#!/usr/bin/env python
# coding: utf-8

# In[34]:


from typing import List, Tuple

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import DistributedType
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import IterableDataset
from torch.distributions import Categorical
import gym 
import numpy as np

from pytorch_lightning.callbacks import Callback
import multiprocessing

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# In[35]:


class CriticNet(nn.Module):
    def __init__(self, obs_size, hidden_size = 100):
        super().__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        value = self.critic(x)
        return value

class ActorNet(nn.Module):
    def __init__(self, obs_size, n_actions, depth, hidden_size = 64):
        super().__init__()

        if depth == 2:
            self.actor = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions),
            )

    def forward(self, x):
        #if x.sum().data.item() == 0:
            #print("ALL ZEROS INPUT : ",self.actor(x))
        logits = self.actor(x)
        logits = torch.nan_to_num(logits)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return dist, action


class ActorCritic():
    def __init__(self, critic, actor):
        self.critic = critic
        self.actor = actor 
    
    @torch.no_grad()
    def __call__(self, state: torch.Tensor):
        dist, action = self.actor(state)
        probs = dist.log_prob(action)
        val = self.critic(state)
        
        return dist, action, probs, val


# In[36]:


class RLDataSet(IterableDataset):
    def __init__(self, batch_maker):
        self.batch_maker = batch_maker
    def __iter__(self):
        return self.batch_maker()


# In[37]:

class PPOLightning(LightningModule):
    
    def __init__(
        self,
        alr,
        clr,
        batch_size,
        clip_eps,
        lamb ,
        epoch_steps,
        gamma,
        depth,
        writer
    ):
        self.writer = writer
        writer = -1
        super().__init__()
        self.save_hyperparameters()

        print("hparams:",self.hparams)
        
        self.env = gym.make("LunarLander-v2")
        self.state = torch.Tensor(self.env.reset())
        self.ep_step = 0
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        print("actions",n_actions)
        
        self.batch_states = []
        self.batch_actions = []
        self.batch_probs = []
        self.batch_advs = []
        self.batch_vals = []
        self.ep_rewards = []
        self.ep_rewards_all = []
        self.ep_vals = []
        self.epoch_rewards = []
        self.avg_reward = 0
        self.avg_ep_reward = 0
        self.last_ep_logged = 0
        
        self.critic = CriticNet(obs_size)
        self.actor = ActorNet(obs_size,n_actions,self.hparams.depth)
        
        self.agent = ActorCritic(self.critic, self.actor)
    
    def forward(self, x):
        
        dist, action = self.actor(x)
        val = self.critic(x)
        
        return dist, action, val
        
    def act_loss(self,state,action,prob_old,adv):
        dist, _ = self.actor(state)
        prob = dist.log_prob(action)
        ratio = torch.exp(prob - prob_old)
        #PPO update
        clip = torch.clamp(ratio, 1 - self.hparams.clip_eps, 1 + self.hparams.clip_eps) * adv
        #negative gradient descent - gradient ascent
        loss = -(torch.min(ratio * adv, clip)).mean()
        return loss
    
    def crit_loss(self,state,val):
        val_new = self.critic(state)
        #MSE
        loss = (val - val_new).pow(2).mean()
        return loss
        
    def compute_gae(self, rewards, values, next_val):
        

        rs = rewards
        vals = values + [next_val]
        
        x = []
        for i in range(len(rs)-1):
            x.append(rs[i]+self.hparams.gamma*vals[i+1] - vals[i])
    
        a = self.compute_reward(x, self.hparams.gamma * self.hparams.lamb)

        return a
    
    def compute_reward(self,rewards, gamma):
        
        rs = []
        sum_rs = 0

        for r in reversed(rewards):
            sum_rs = (sum_rs * gamma) + r
            rs.append(sum_rs)


        return list(reversed(rs))

    
    def make_batch(self):
        for i in range(self.hparams.epoch_steps):

            _, action, probs, val = self.agent(self.state)
            next_state, reward, done, _ = self.env.step(action.item())
            self.ep_step += 1
            
            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_probs.append(probs)
            self.ep_rewards.append(reward)
            self.ep_vals.append(val.item())

            self.state = torch.Tensor(next_state)
            
            end = i == (self.hparams.epoch_steps -1)

            if done or end:
                
                if end and not done:
                    #if epoch ends before terminal state, bootstrap value
                    with torch.no_grad():
                        #print("epoch ended early")
                        _,_,_,val = self.agent(self.state)
                        next_val = val.item()
                else:
                    next_val = 0
                
                #compute batch discounted rewards
                self.ep_rewards.append(next_val)
                self.batch_vals += self.compute_reward(self.ep_rewards,self.hparams.gamma)[:-1]
                self.batch_advs += self.compute_gae(self.ep_rewards,self.ep_vals, next_val)
                
                self.epoch_rewards.append(sum(self.ep_rewards))
                #print("Total for Ep :",sum(self.ep_rewards))
                self.ep_rewards_all.append(sum(self.ep_rewards))
                self.ep_rewards.clear()
                self.ep_vals.clear()
                self.ep_step = 0
                self.state = torch.Tensor(self.env.reset())
                
            if end:
                data = zip(self.batch_states,
                            self.batch_actions,
                            self.batch_probs,
                            self.batch_vals,
                            self.batch_advs)

                for (s, a, p, v, ad) in data:
                    yield s, a, p, v, ad
                    
                #logs
                self.avg_ep_reward = sum(self.epoch_rewards)/len(self.epoch_rewards)
                self.epoch_rewards.clear()
                
                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_probs.clear()
                self.batch_vals.clear()
                self.batch_advs.clear()
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        
        state,action,prob_old,val,adv = batch

        # normalize adv
        adv = (adv - adv.mean())/adv.std()
        
        for i in range(self.last_ep_logged,len(self.ep_rewards_all)):
             self.log("ep_reward",self.ep_rewards_all[i],prog_bar=True, on_step=False, on_epoch=True, logger=True)
             self.last_ep_logged += 1

        self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        
        if optimizer_idx == 0:
            loss = self.act_loss(state, action, prob_old, adv)
            self.log('act_loss', loss, on_step=False, on_epoch=True, prog_bar=True,logger=True)

            self.writer.writerow([self.global_step, self.avg_ep_reward, loss.unsqueeze(0).item()])

            return loss

        elif optimizer_idx == 1:
            loss = self.crit_loss(state,val)
            self.log('crit_loss', loss, on_step=False, on_epoch=True, prog_bar=True,logger=True)

            self.writer.writerow([self.global_step, self.avg_ep_reward, loss.unsqueeze(0).item()])

            return loss

    
    def configure_optimizers(self) -> List[Optimizer]:
        a_opt = optim.Adam(self.actor.parameters(), lr=self.hparams.alr)
        c_opt = optim.Adam(self.critic.parameters(), lr=self.hparams.clr)
        return a_opt,c_opt
    
    def __dataloader(self):
        dataset = RLDataSet(self.make_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader
    
    def train_dataloader(self):
        return self.__dataloader()


# In[38]:

class ReturnCallback(Callback):
    def __init__(self ):
        self.total = []

    def on_train_epoch_end(self, trainer, pl_module):
        print("Callback")
        self.total.append(trainer.callback_metrics['avg_ep_reward'].item())
    
    def get_total(self):
        return self.total

from pathlib import Path
import csv
import os

def pickFileName():

    Path("log/trainingvalsPPO/").mkdir(parents=True, exist_ok=True)

    files = os.listdir('log/trainingvalsPPO/')

    return '{}.csv'.format(len(files)+1)

num_epochs=25000


f = open('log/trainingvalsPPO/{}'.format(pickFileName()), 'w+')
writer = csv.writer(f)

model = PPOLightning(
        6.99e-4,#alr,
        7.07e-4,#clr,
        64,#batch_size,
        0.208,#clip_eps,
        0.953,#lamb,
        2048, #epoch steps
        0.99, #gamma
        2,#depth,
        writer
    )

tb_logger = TensorBoardLogger("log/")

trainer = Trainer(
        gpus=0,
        max_epochs=num_epochs,
        logger=tb_logger)

trainer.fit(model)

print("finished training")

