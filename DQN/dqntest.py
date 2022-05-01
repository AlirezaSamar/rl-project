from logging import exception
import gym
import torch as T
from dqnagent import DQNetwork
from dqntrain import avgRewards
import matplotlib.pyplot as plt
import numpy as np
import sys

class testAgent:
    def __init__(self, filename, inputDims, nActions):
        self.net = DQNetwork(1, nActions=nActions, 
            inputDims=inputDims)

        self.net.load_state_dict(T.load(filename))
        self.net.eval()

    def choose_action(self, observation):
        state = T.tensor(np.array(observation)).to(self.net.device)
        with T.no_grad():
            actions = self.net.forward(state)
        action = T.argmax(actions).item()
        return action

def testAnAgent(agent, ix, render=False, nEpisodes = 100):
    env = gym.make('LunarLander-v2')
    scores = []
    animation = "|/-\\"

    for i in range(nEpisodes):    
        score = 0
        terminal = False
        observation = env.reset()
        while not terminal:
            if render and i%20==0:
                env.render()

            action = agent.choose_action(observation)
            observation_, reward, terminal, _ = env.step(action)
            score += reward
            observation = observation_
        
        scores.append(score)
        avgScore = np.mean(scores[-100:])

        if render:
            print(f"epsiode {i}, score {score:.2f}, avg score {avgScore:.2f}.")
        else:
            print("Testing [" + animation[i % len(animation)] +"]", end="\r")

    print(f" [{ix}] Avg Score after {nEpisodes} eps: {avgScore:.2f}.")

    return scores

def plotAgentScores(scores):
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.ylim((0, 300))
    plt.show()

if __name__ == '__main__':
    maxEpisodes = 100

    try:
        filename = sys.argv[1]
        agent = testAgent(filename, inputDims=[8], nActions=4) #ignore perams
    except FileNotFoundError:
        print("File cant be found mate")
        exit(0)
    except Exception as e:
        print(e)
        exit(0)

    nAgents = 1

    agentScores = []
    for i in range(nAgents):
        scores = testAnAgent(agent, render=nAgents==1, nEpisodes=maxEpisodes, ix=i)
        agentScores.append(scores)

    plotAgentScores(avgRewards(agentScores))