import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
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
axis[0].set_xlim(0, ln)
axis[0].set_ylim(res.min(), res.max())

line2, = axis[1].plot([],[])
axis[1].set_xlabel("Episodes")
axis[1].set_ylabel("Average Episode Length")
axis[1].set_title("Episode Length")
axis[1].set_xlim(0, ln)
axis[1].set_ylim(lens.min(), lens.max())

line3, = axis[2].plot([],[])
axis[2].set_xlabel("Episodes")
axis[2].set_ylabel("Average Loss")
axis[2].set_title("Loss")
axis[2].set_xlim(0, ln)
axis[2].set_ylim(loss.min(),loss.max())

line = [line1,line2,line3]

def init():
    line[0].set_data([],[])
    line[1].set_data([],[])
    line[2].set_data([],[])

    return line

figure.tight_layout()

def append_to_line(line, x, y):
    xd, yd = [list(t) for t in line.get_data()]
    xd.append(x)
    yd.append(y)
    line.set_data(xd, yd)
    print(yd)

def animate(dat):
    i, r, le, lo = dat

    append_to_line(line[0], i, r)
    append_to_line(line[1], i, le)
    append_to_line(line[2], i, lo)

FFwriter = matplotlib.animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
ani = matplotlib.animation.FuncAnimation(figure, animate, frames=gendat, interval=10, repeat=False, save_count=ln)

#plt.show()

ani.save("results.mp4",writer=FFwriter, dpi=500)
