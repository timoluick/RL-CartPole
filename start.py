import gym
from torch import nn
import torch
from collections import deque
import copy
import random
import matplotlib.pyplot as plt
from IPython import display

env = gym.envs.make('CartPole-v0')


class Network(nn.Module):
    def __init__(self, in_size, out_size):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(in_features=in_size, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=out_size)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.linear1(x))
        x = torch.nn.functional.tanh(self.linear2(x))
        return x


class Agent:
    def __init__(self):
        torch.manual_seed(1423)
        self.policy_net = Network(in_size=4, out_size=2)
        self.target_net = copy.deepcopy(self.policy_net)
        self.epsilon = 1
        self.max_replay_size = 256
        self.replay = deque(maxlen=self.max_replay_size)
        self.action_space_len = env.action_space.n
        self.gamma = torch.tensor(0.95).float()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

        self.sync_counter = 0
        self.sync = 5

    def make_step(self, state):
        if torch.rand(1,).item() > self.epsilon:
            with torch.no_grad():
                action = self.policy_net(torch.tensor(state).float())
            _, action = torch.max(action, 0)
        else:
            action = torch.randint(0, self.action_space_len, (1,))
        return action.item()

    def get_q_next(self, state):
        with torch.no_grad():
            action = self.policy_net(torch.tensor(state).float())
        _, action = torch.max(action, 1)
        return action

    def add_experience(self, experience):
        self.replay.append(experience)

    def sample_from_experience(self, sample_size):
        sample = random.sample(self.replay, sample_size)
        states = torch.tensor([exp[0] for exp in sample]).float()
        actions = torch.tensor([exp[1] for exp in sample]).float()
        rewards = torch.tensor([exp[2] for exp in sample]).float()
        next_states = torch.tensor([exp[3] for exp in sample]).float()
        return states, actions, rewards, next_states

    def train(self, batch_size):
        if len(self.replay) < batch_size:
            return
        s, a, r, sn = self.sample_from_experience(batch_size)
        if self.sync_counter == self.sync:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.sync_counter = 0
            #print(self.policy_net(s[0]))
            #print(self.target_net(s[0]))
            #print('-----------')

        qp = self.policy_net(s)
        pred_return, actions = torch.max(qp, 1)
        print(actions)

        q_next = self.get_q_next(sn)
        target_return = r + self.gamma * q_next

        self.optimizer.zero_grad()
        loss = self.loss(pred_return, target_return)
        loss.backward()
        self.optimizer.step()

        self.sync_counter += 1

        return loss.item()


def plot_durations(x, y):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('x')
    plt.ylabel('y')

    lines = plt.plot(x, y)
    plt.setp(lines[0], linewidth=1)

    plt.pause(0.0001)
    display.clear_output(wait=True)
    display.display(plt.gcf())


agent = Agent()

episodes = 10000

episodes_list = []
episode_length = []

index = 0

for i in range(episodes):
    observation, done = env.reset(), False
    length = 0
    while not done:
        length += 1
        index += 1
        action = agent.make_step(observation)
        observation_next, reward, done, _ = env.step(action)
        agent.add_experience([observation, action, reward, observation_next])
        observation = observation_next
        agent.train(16)
        env.render()
    episodes_list.append(i)
    episode_length.append(length)
    if i % 20 == 0:
        plot_durations(episodes_list, episode_length)
    if agent.epsilon > 0.05:
        agent.epsilon -= 1/5000
