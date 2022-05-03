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
    ln += len(a)

ln = round(ln/len(avg_rewards))
xw = list(range(ln))

res = np.empty((0,ln))
lens = np.empty((0,ln))
loss = np.empty((0,ln))

for a,b,c in zip(avg_rewards,avg_lens,avg_losss):
    a = zoom(a,ln/len(a))
    b = zoom(b,ln/len(b))
    c = zoom(c,ln/len(c))

    res = np.insert(res,0,a,axis=0)
    lens = np.insert(lens,0,b, axis=0)
    loss = np.insert(loss,0,c, axis=0)

res = np.mean(res,axis=0)
lens = np.mean(lens,axis=0)
loss = np.mean(loss,axis=0)

randres = []
randlens = []
with open(os.path.join(dirn,os.pardir,'results.txt')) as f:
    lines = f.readlines()

    for i,line in enumerate(lines):
        line = line.split(',')
        randres.append(float(line[0]))
        randlens.append(float(line[1]))
        if i >= ln+48:
            break

randres = np.convolve(randres, np.ones(50)/50, mode='valid')
randlens = np.convolve(randlens, np.ones(50)/50, mode='valid')

figure, axis = plt.subplots(1,3, figsize=(15,5))

algo = dirn.split('/')[-1]

axis[0].plot(xw,res, label=algo)
axis[0].plot(xw,randres, color='r', label="Random Agent")
axis[0].set_xlabel("Episodes")
axis[0].set_ylabel("Average Reward")
axis[0].set_title("Reward")
axis[0].legend()

axis[1].plot(xw,lens,  label=algo)
axis[1].plot(xw,randlens, color='r', label="Random Agent")
axis[1].set_xlabel("Episodes")
axis[1].set_ylabel("Average Episode Length")
axis[1].set_title("Episode Length")
axis[1].legend()

axis[2].plot(xw,loss,  label=algo)
axis[2].set_xlabel("Episodes")
axis[2].set_ylabel("Average Loss")
axis[2].set_title("Loss")
axis[2].legend()
figure.tight_layout()

plt.savefig("results.png",transparent=True, dpi=500)




