import os
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from tensorflow.python.summary.summary_iterator import summary_iterator
from scipy.ndimage import zoom
import matplotlib

x= os.walk("/home/scrungus/Documents/code/rl-project/PPO/runs")

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

def gendat():
    for i, (r, le, lo) in enumerate(zip(res,lens,loss)):
        yield i, r, le , lo


figure, axis = plt.subplots(1,3, figsize=(15,5))

line1, = axis[0].plot([],[])
axis[0].set_xlabel("Episodes")
axis[0].set_ylabel("Average Reward")
axis[0].set_title("Reward")

line2, = axis[1].plot([],[])
axis[1].set_xlabel("Episodes")
axis[1].set_ylabel("Average Episode Length")
axis[1].set_title("Episode Length")

line3, = axis[2].plot([],[])
axis[2].set_xlabel("Episodes")
axis[2].set_ylabel("Average Loss")
axis[2].set_title("Loss")

line = [line1,line2,line3]

def init():
    line[0].set_data([],[])
    line[1].set_data([],[])
    line[2].set_data([],[])

    return line

figure.tight_layout()

def animate(dat):

    i,r, le , lo = dat

    print("got",i,r,le,lo)
    line[0].set_data(i,r)
    line[1].set_data(i,le)
    line[2].set_data(i,lo)

    return line

FFwriter = matplotlib.animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
ani = matplotlib.animation.FuncAnimation(figure, animate, init_func=init, frames=gendat, interval=20, repeat=False, blit=True)

plt.show()

ani.save("results.mp4",writer=FFwriter, dpi=500)
#plt.savefig("results.png",transparent=True, dpi=500)




