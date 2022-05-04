import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ReplayBuffer:
    def __init__(self, inputDims, memSize):
        self.memSize = memSize
        self.memCntr = 0

        self.stateMemory = np.zeros((self.memSize, *inputDims), dtype=np.float32)
        self.newStateMemory = np.zeros((self.memSize, *inputDims), dtype=np.float32)
        self.actionMemory = np.zeros(self.memSize, dtype=np.int32)
        self.rewardMemory = np.zeros(self.memSize, dtype=np.float32)
        self.terminalMemory = np.zeros(self.memSize, dtype=bool)

    def storeTransition(self, state, action, reward, state_, terminal):
        index = self.memCntr % self.memSize
        self.memCntr += 1

        self.stateMemory[index] = state
        self.newStateMemory[index] = state_
        self.rewardMemory[index] = reward
        self.actionMemory[index] = action
        self.terminalMemory[index] = terminal

    def sample(self, batchSize, device):
        maxMem = min(self.memCntr, self.memSize)
        batch = np.random.choice(maxMem, batchSize, replace=False)

        stateBatch = T.tensor(self.stateMemory[batch]).to(device)
        newStateBatch = T.tensor(self.newStateMemory[batch]).to(device)
        rewardBatch = T.tensor(self.rewardMemory[batch]).to(device)
        terminalBatch = T.tensor(self.terminalMemory[batch]).to(device)

        actionBatch = self.actionMemory[batch]

        return stateBatch, actionBatch, newStateBatch, rewardBatch, terminalBatch
    

class DQNetwork(nn.Module):
    def __init__(self, lr, inputDims, nActions, 
            fc1Dims=256, fc2Dims=256, device='cpu'):
        super(DQNetwork, self).__init__()

        self.fcls = nn.Sequential(
            nn.Linear(*inputDims, fc1Dims),
            nn.ReLU(),
            nn.Linear(fc1Dims, fc2Dims),
            nn.ReLU(),
            nn.Linear(fc2Dims, nActions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, state):
        actionValues = self.fcls(state)

        return actionValues


class Agent():
    def __init__(self, gamma, epsilon, lr,  inputDims, batchSize, nActions,
            c, k, maxMemorySize=100000, epsMin=0.01, epsDecr=5e-5):
        self.gamma = gamma # decay rate
        self.epsilon = epsilon # exploration rate
        self.epsMin = epsMin # min exp rate
        self.epsDecr = epsDecr # rate of exp decrease
        self.lr = lr # learning rate
        self.c = c # soft update rate
        self.k = k # batch sampling rate

        self.action_space = np.arange(nActions)
        self.batchSize = batchSize

        self.device = self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.localNet = DQNetwork(self.lr, nActions=nActions, 
            inputDims=inputDims, device=self.device)
        self.targetNet = DQNetwork(self.lr, nActions=nActions, 
            inputDims=inputDims, device=self.device)
        
        self.replayMemory = ReplayBuffer(inputDims, maxMemorySize)

    def choose_action(self, observation):
        if np.random.rand() > self.epsilon:
            state = T.tensor(np.array(observation)).to(self.device)
            with T.no_grad():
                actions = self.localNet(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action

    def learn(self):
        if self.replayMemory.memCntr < self.batchSize:
            return

        if self.replayMemory.memCntr % self.k == 0:
            self.localNet.optimizer.zero_grad()

            batchIndexes = np.arange(self.batchSize, dtype=np.int32)
            (stateBatch, actionBatch, newStateBatch, rewardBatch,
            terminalBatch) = self.replayMemory.sample(self.batchSize, self.device)
            
            qLocal = self.localNet(stateBatch)[batchIndexes, actionBatch]
            with T.no_grad():
                qNext = self.targetNet(newStateBatch).detach()
            qNext[terminalBatch] = 0.0

            qTarget = rewardBatch + self.gamma * T.max(qNext, dim=1)[0]

            loss = self.localNet.loss(qTarget, qLocal).to(self.device)
            loss.backward()
            self.localNet.optimizer.step()

        if self.replayMemory.memCntr % self.c == 0:
            self.targetNet.load_state_dict(self.localNet.state_dict())
        
        self.epsilon = max(self.epsilon-self.epsDecr, self.epsMin)

        return loss.item()

    def save(self, filename):
        T.save(self.localNet.state_dict(), filename)