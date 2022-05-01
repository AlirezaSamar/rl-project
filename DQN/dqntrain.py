import gym
from dqnagent import Agent
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

def trainAnAgent(agent, displayScores=False, clip=False, render=False, save=False, nEpisodes = 270):
    writer = SummaryWriter()
    env = gym.make('LunarLander-v2')
    scores = []
    epLengths  = []
    animation = "|/-\\"
    startTime = time.process_time()

    for i in range(nEpisodes):    
        score = 0
        epLen = 0
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
            observation = observation_
            epLen += 1 
        
        scores.append(score)
        epLengths.append(epLen)
        avgScore = np.mean(scores[-100:])
        avgEpLen = np.mean(epLengths[-100:])

        writer.add_scalar("avg_reward",avgScore,i)
        writer.add_scalar("avg_len",avgEpLen,i)
        writer.add_scalar("loss",loss,i)

        if displayScores:
            print(f"epsiode {i}, score {score:.2f}, avg score {avgScore:.2f}, epsilon {agent.epsilon:.3f}")
        else:
            print("Training [" + animation[i % len(animation)] +"]", end="\r")
        if clip and avgScore >= 200:
            nEpisodes = i
            writer.close()
            break

    runTime = time.process_time() - startTime
    print(f"\n Training took {runTime:.2f} seconds for {nEpisodes} episodes.",
          f"\n Transitions encountered: {agent.replayMemory.memCntr},",
          f" Last 100 avg: {avgScore:.2f}")

    if save:
        filename = input("Enter filename to save agent... ")
        if filename != "no":
            agent.save(filename)

    return runTime


if __name__ == '__main__':
    nAgents = 1
    agentScores = []
    totalTime = 0
    maxEpisodes = 400

    clip=nAgents==1
    saveAgent=False
    saveScores=False
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


    # lrs = [.0001, .0005, .00025]

    # for lr in lrs:
    #     for _ in range(nAgents):
    #         agent = Agent(gamma=gamma, epsilon=epsilon, epsMin=epsMin, epsDecr=epsDecr,
    #             batchSize=batchSize, nActions=4, inputDims=[8], lr=lr, c=c, k=k)

    #         scores, runTime = trainAnAgent(agent, clip=clip, render=render, 
    #             displayScores=displayScores, save=saveAgent, nEpisodes=maxEpisodes)
            
    #         totalTime += runTime
    #         agentScores.append(scores)
    #     writeScores(avgRewards(agentScores), filename=f"lrIS{lr}")

    for _ in range(nAgents):
        agent = Agent(gamma=gamma, epsilon=epsilon, epsMin=epsMin, epsDecr=epsDecr,
            batchSize=batchSize, nActions=4, inputDims=[8], lr=lr, c=c, k=k)

        runTime = trainAnAgent(agent, clip=clip, render=render, 
            displayScores=displayScores, save=saveAgent, nEpisodes=maxEpisodes)
        
        totalTime += runTime