import gym
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import torch.optim as optim

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


def optimize():
    env = gym.make("MountainCar-v0")
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    net = ActorCritic(num_inputs, num_outputs, hidden_size)
    optimizer = optim.Adam(net.parameters(), 1e-3)  # 200 episode solved

    # optimizer =
    for episode in range(episodes):
        rewards_entropy = []
        rewards = []
        targets = []
        V_ss = []
        baselines = []
        log_pi_sa = []
        # entropys = []
        state = env.reset()
        done = False
        while not done:
            # env.render()
            value, policy = net.forward(state)  # s(t）
            V_ss.append(value)
            # print(value.requires_grad)
            pi = policy[0].detach().numpy()
            action = np.random.choice(num_outputs, p=pi)
            # print(type(action))

            state, reward, done, info = env.step(action)    # r(t)

            v, _ = net.forward(state)  # s(t+1)
            target = reward + v
            # print(target.requires_grad) true
            targets.append(target)
            entropy = -torch.sum(policy[0] * torch.log(policy[0]))
            reward_entropy = reward + entropy

            rewards.append(reward)
            rewards_entropy.append(reward_entropy)
            baselines.append(value)
            # print(torch.log(policy[0][action]).requires_grad)
            log_pi_sa.append(torch.log(policy[0][action]))

        returns = calc_returns(rewards)
        # print(returns.requires_grad)
        returns = torch.Tensor(returns)
        targets = torch.cat(targets).squeeze()
        V_ss = torch.cat(V_ss).squeeze()    # ******
        # print(targets.requires_grad)
        # print(returns)

        baselines = torch.cat(baselines).squeeze()
        # print(baselines.requires_grad)
        adv = returns - baselines
        # print(adv.requires_grad)
        log_pi_sa = torch.stack(log_pi_sa)
        # print(log_pi_sa.requires_grad)
        td_error = targets - V_ss
        pg_loss = log_pi_sa * td_error
        # v_loss = F.mse_loss(baselines, returns)
        v_loss = F.mse_loss(targets, V_ss)
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
            print(targets)
            print(episode, sum(rewards))

        if episode % 1000 == 0:
            state = env.reset()
            done = False
            while not done:
                env.render()
                value, policy = net.forward(state)
                action = np.random.choice(num_outputs, p=policy[0].detach().numpy())

                state, reward, done, info = env.step(action)


if __name__ == '__main__':
    optimize()

