import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def gendat():

    res = list(np.random.random_sample(size=609))
    lens = list(np.random.random_sample(size=609))
    loss = list(np.random.random_sample(size=609))
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




