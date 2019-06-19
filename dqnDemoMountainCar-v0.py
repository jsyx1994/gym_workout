import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
steps_done = 0


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:  # 如果空间足够，那么预留一个可以索引的位置给新元素
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(DQN, self).__init__()

        self.num_actions = num_actions
        self.fc1 = nn.Linear(num_inputs, hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.head = nn.Linear(hidden_size, num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.Tensor(torch.from_numpy(x).float())
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    # plt.ylabel('Duration')
    plt.ylabel('Position')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated



env = gym.make('MountainCar-v0')
# Get number of actions from gym action space
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.n
policy_net = DQN(num_inputs,num_outputs, 64)
target_net = DQN(num_inputs,num_outputs, 64)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# policy_net([1.,2.,3.,4.])

optimizer = optim.Adam(policy_net.parameters(),1e-3)
memory = ReplayMemory(10000)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print(policy_net(state).max(0)[1])
            # print(policy_net.forward(state))
            return policy_net(state).max(0)[1]   #  返回最大值的下标
    else:
        # random.randrange()使用均匀分布随机采样
        return torch.tensor([[random.randrange(num_outputs)]], dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # print(transitions)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # print(batch)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                       batch.next_state)),  dtype=torch.uint8)

    non_final_next_states = np.asarray(batch.next_state)
    # print(non_final_next_states.shape)

    # print(non_final_next_states)

    # print(np.asarray(batch.state))
    # print(batch.action)
    state_batch = np.asarray(batch.state)
    action_batch = torch.LongTensor(batch.action)
    reward_batch = torch.Tensor(batch.reward)
    # print(reward_batch)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(action_batch.size())
    # print(policy_net.forward(state_batch).squeeze(0).size())
    # print(policy_net.forward(state_batch))
    state_action_values = policy_net.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # next_state_values = torch.zeros(BATCH_SIZE)

    # Q(s,a) = r + maxQ(s_,a_) for all a_ in state s_
    # print(target_net(non_final_next_states))
    next_state_values = target_net(non_final_next_states).max(1)[0].detach()
    # print(next_state_values)
    # Compute the expected Q values
    # print(reward_batch.shape)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # print(expected_state_action_values)
    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)
    # print(state_action_values.size(),expected_state_action_values.size())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 5000

episode_durations = []
positions = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    position = state[0]
    rewards = []
    for t in count():
        # Select and perform an action
        action = select_action(state).item()
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = - reward
        if next_state[0] > position:
            position = next_state[0]
        rewards.append(reward)
        # print(reward)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            print(str(i_episode) + ' rewards', sum(rewards))
            episode_durations.append(position)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        # print('yes')
    if i_episode % 100 == 0:
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = select_action(state).item()
            state, reward, done, info = env.step(action)

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()