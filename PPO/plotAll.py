import os
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from tensorflow.python.summary.summary_iterator import summary_iterator
from scipy.ndimage import zoom

dirn = os.path.dirname(__file__)
x= os.walk(dirn+"/runs")

avg_rewards = []
avg_lens = []
avg_losss = []
for folder, _, fle in x:
    for f in fle:
        avg_reward = []
        avg_len = []
        avg_loss = []
        for e in summary_iterator(folder+'/'+f):
            for v in e.summary.value:
                if v.tag == 'avg_reward':
                    avg_reward.append(v.simple_value)
                if v.tag == 'avg_len':
                    avg_len.append(v.simple_value)
                if v.tag == 'loss':
                    avg_loss.append(v.simple_value)
        avg_rewards.append(avg_reward)
        avg_lens.append(avg_len)
        avg_losss.append(avg_loss)

ln = 0

for a in avg_rewards:
    if len(a) > ln:
        ln = len(a)

xw = list(range(ln))

for i in range(len(avg_rewards)):
    if len(avg_rewards[i]) < ln:
        avg_rewards[i] += [None]*(ln-len(avg_rewards[i]))
        avg_lens[i] += [None]*(ln-len(avg_lens[i]))
        avg_losss[i] += [None]*(ln-len(avg_losss[i]))

figure, axis = plt.subplots(1,3, figsize=(15,5))

algo = dirn.split('/')[-1]

for res, lens, loss in zip(avg_rewards,avg_lens,avg_losss):
    axis[0].plot(xw,res)
    axis[1].plot(xw,lens)
    axis[2].plot(xw,loss)

axis[0].set_xlabel("Episodes")
axis[0].set_ylabel("Average Reward")
axis[0].set_title("Reward")


axis[1].set_xlabel("Episodes")
axis[1].set_ylabel("Average Episode Length")
axis[1].set_title("Episode Length")


axis[2].set_xlabel("Episodes")
axis[2].set_ylabel("Average Loss")
axis[2].set_title("Loss")

figure.tight_layout()

plt.savefig("results.png",transparent=True, dpi=500)




