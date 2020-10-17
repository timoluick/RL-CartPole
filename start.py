import gym
import torch
from collections import deque
from copy import deepcopy
import copy
import random
import matplotlib.pyplot as plt
from IPython import display


class DQN(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(DQN, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=in_size, out_features=20)
        self.linear2 = torch.nn.Linear(in_features=20, out_features=out_size)
        #self.linear3 = torch.nn.Linear(in_features=30, out_features=out_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        #x = torch.relu(self.linear3(x))
        return x


class Agent:
    def __init__(self, in_size, out_size):
        self.memory_replay_size = 200
        self.memory = deque(maxlen=self.memory_replay_size)

        self.policy_net = DQN(in_size=in_size, out_size=out_size)
        self.target_net = deepcopy(self.policy_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss = torch.nn.MSELoss()
        self.sync_counter = 0
        self.sync = 10
        self.gamma = torch.tensor([0.95]).float()
        self.epsilon = 1
        self.min_epsilon = 0.05

    def add_experience(self, experience):
        self.memory.append(experience)

    def get_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                action = self.policy_net(state).max(0)[1].item()
            return action
        else:
            return random.randint(0, 1)

    def get_sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        n_s = torch.tensor([exp[1] for exp in sample]).float()
        r = torch.tensor([exp[2] for exp in sample]).float()
        a = torch.tensor([exp[3] for exp in sample]).float()
        d = torch.tensor([exp[4] for exp in sample]).float()
        return s, n_s, r, a, d

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        if self.sync_counter == self.sync:
            self.sync_counter = 0
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.sync_counter += 1

        self.optimizer.zero_grad()

        s, n_s, r, a, d = self.get_sample(batch_size)

        q_next = self.target_net(n_s).max(1)[0]
        t_return = r + torch.mul(q_next * self.gamma, (~d.bool()).float())
        current = self.policy_net(s)
        target = current.clone()

        for i in range(len(target)):
            target[i][a[i].long()] = t_return[i]

        target = current + self.gamma * (target - current)

        loss = self.loss(current, target)
        loss.backward()
        self.optimizer.step()


def plot(x, y):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('x')
    plt.ylabel('y')

    lines = plt.plot(x, y)
    plt.setp(lines[0], linewidth=2)

    plt.pause(0.0001)
    display.clear_output(wait=True)
    display.display(plt.gcf())


env = gym.envs.make('CartPole-v1')

env.reset()

epochs = 3000

agent = Agent(in_size=4, out_size=2)

i = 0

epoch_lengths = []

for epoch in range(epochs):
    state, done = env.reset(), False
    length = 0
    while not done:
        length += 1

        action = agent.get_action(torch.from_numpy(state).float())
        next_state, reward, done, _ = env.step(action)
        agent.add_experience([state, next_state, reward, action, done])
        state = next_state
        agent.train(16)
        env.render()
    epoch_lengths.append(length)
    agent.epsilon = max(agent.min_epsilon, agent.epsilon-1/2000)
    if epoch % 10 == 0:
        plot(range(epoch + 1), epoch_lengths)
