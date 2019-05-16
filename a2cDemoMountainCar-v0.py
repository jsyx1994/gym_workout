import gym
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
hidden_size = 32
episodes = 3000
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'baseline'))
gamma = .999

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.shared = nn.Linear(num_inputs, hidden_size)

        self.critic_linear1 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear1 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        # print(state.size())
        base = F.relu(self.shared(state))

        value = self.critic_linear1(base)
        value = self.critic_linear2(value)

        policy_dist = self.actor_linear1(base)
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist


def calc_returns(rewards: list):
    # returns = torch.zeros(len(rewards))
    returns = [0 for _ in range(len(rewards))]
    r = 0
    for i in range(len(rewards) - 1, -1, -1):
        returns[i] = rewards[i] + gamma * r
        r = returns[i]
    return returns

def plot(y):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(y, dtype=torch.float)
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

def optimize():
    env = gym.make("MountainCar-v0")
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    net = ActorCritic(num_inputs, num_outputs, hidden_size)
    optimizer = optim.Adam(net.parameters(), 1e-3)

    # optimizer =
    positions = []
    for episode in range(episodes):
        rewards_entropy = []
        rewards = []
        baselines = []
        log_pi_sa = []

        # entropys = []
        state = env.reset()
        position = state[0]
        # print(position)
        done = False
        while not done:
            # env.render()
            value, policy = net.forward(state)
            # print(value.requires_grad)
            pi = policy[0].detach().numpy()

            action = np.random.choice(num_outputs, p=pi)
            # print(type(action))

            state, reward, done, info = env.step(action)
            if state[0] > position:
                position = state[0]

            entropy = -torch.sum(policy[0] * torch.log(policy[0]))
            reward_entropy = reward + entropy

            rewards.append(reward)
            rewards_entropy.append(reward_entropy)
            baselines.append(value)
            # print(torch.log(policy[0][action]).requires_grad)
            log_pi_sa.append(torch.log(policy[0][action]))

        positions.append(position)
        plot(positions)
        returns = calc_returns(rewards)
        # print(returns.requires_grad)
        returns = torch.Tensor(returns)
        # print(returns)

        baselines = torch.cat(baselines).squeeze()
        # print(baselines.requires_grad)
        adv = returns - baselines
        # print(adv.requires_grad)
        log_pi_sa = torch.stack(log_pi_sa)
        # print(log_pi_sa.requires_grad)

        pg_loss = log_pi_sa * adv
        v_loss = F.mse_loss(baselines, returns)
        # print(v_loss)
        # print(v_loss.requires_grad)
        # v_loss.backward()
        # 需要加符号，因为这是我们最大化return后的导数，需要梯度上升更新。torch默认因该是为梯度下降法

        ac_loss = - pg_loss.mean() + v_loss  # + 0.001 * entropy_term
        optimizer.zero_grad()
        ac_loss.backward()
        optimizer.step()
        # print(baselines)

        if episode % 10 == 0:
            print(baselines)
            print(episode, sum(rewards))

        if episode % 100 == 0:
            state = env.reset()
            done = False
            while not done:
                env.render()
                value, policy = net.forward(state)
                action = np.random.choice(num_outputs, p=policy[0].detach().numpy())

                state, reward, done, info = env.step(action)


if __name__ == '__main__':
    optimize()

