import gym
from dqnagent import Agent
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter

def trainAnAgent(agent, displayScores=False, clip=False, render=False, save=False, nEpisodes = 270):
    writer = SummaryWriter()
    env = gym.make('LunarLander-v2')
    scores = []
    epLengths  = []
    losses = []
    animation = "|/-\\"

    for i in range(nEpisodes):    
        score = 0
        epLen = 0
        runningLoss = 0.0
        terminal = False
        observation = env.reset()
        while not terminal:
            if render and i%25==0:
                env.render()

            action = agent.choose_action(observation)
            observation_, reward, terminal, _ = env.step(action)
            score += reward
            agent.replayMemory.storeTransition(observation, action, reward, observation_, terminal)
            loss = agent.learn()
            if loss is not None:
                runningLoss += loss
            observation = observation_
            epLen += 1 
        
        scores.append(score)
        epLengths.append(epLen)
        losses.append(runningLoss)
        avgScore = np.mean(scores[-100:])
        avgEpLen = np.mean(epLengths[-100:])
        avgLoss = np.mean(losses[-100:])

        writer.add_scalar("avg_reward",avgScore,i)
        writer.add_scalar("avg_len",avgEpLen,i)
        writer.add_scalar("loss",runningLoss,i)

        if displayScores:
            print(f"epsiode {i}, score {score:.2f}, avg score {avgScore:.2f}, epsilon {agent.epsilon:.3f}")
        else:
            print("Training [" + animation[i % len(animation)] +"]", end="\r")
        if clip and avgScore >= 200:
            nEpisodes = i
            break

    print(f" {nEpisodes} episodes, {agent.replayMemory.memCntr} transitions encountered,",
          f"\n Last 100 avg: {avgScore:.2f}, Latest Loss: {runningLoss}")
    writer.close()

    if save:
        filename = input("Enter filename to save agent... ")
        if filename != "no":
            agent.save(filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", required=False, help="The number of agents to train")
    args = parser.parse_args()

    try:
        argAgents = int(args.agents) if args.agents else 1
    except:
        print("Warning '-a' argument must be an integer")
        exit()
    nAgents = argAgents

    agentScores = []
    totalTime = 0
    maxEpisodes = 5000

    clip=True
    saveAgent=False
    render=False
    displayScores=False

    gamma=.99 # decay rate
    epsilon=1.0 # exploration rate
    epsMin=.01 # min exp rate
    epsDecr=1e-4 # rate of exp decrease
    batchSize=64 
    lr=.0005 # learning rate
    c=4 # target network update rate
    k=1 # batch sampling rate

    for _ in range(nAgents):
        agent = Agent(gamma=gamma, epsilon=epsilon, epsMin=epsMin, epsDecr=epsDecr,
            batchSize=batchSize, nActions=4, inputDims=[8], lr=lr, c=c, k=k)

        trainAnAgent(agent, clip=clip, render=render, 
            displayScores=displayScores, save=saveAgent, nEpisodes=maxEpisodes)
        
