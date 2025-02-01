import torch
import torch.nn as nn
import gymnasium as gym

from collections import namedtuple, deque
import random
from tqdm import tqdm

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Buffer(object):

    def __init__(self, capacity=10):
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def __len__(self):
        return len(self.buffer)


class DQNAgent(object):
    def __init__(
            self,
            env: gym.Env,
            net: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            lr: float,
            eps: float = 0.01,
            gamma: float = 0.99,
            buffer_capacity: int = 10,
            ):
        self.env = env
        self.net = net
        self.optim = optimizer(net.parameters(), lr=lr)
        self.device = device
        self.eps = eps
        self.gamma = gamma
        self.buffer = Buffer(capacity=buffer_capacity)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        sample = random.random()
        if sample < self.eps:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long, device=self.device)
        else:
            # don't accumulate gradients here.
            # only compute them when evaluating Q(s_t, a_t) from states in the buffer
            with torch.no_grad():
                return self.net(state).max(1).indices.view(1, 1)

    def step(self, batch_size):
        transitions = self.buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # this selects the Q(s, a) values corresponding to states seen and actions taken
        # the gather function selects the value Q(s_t, a) from the row [Q(s_t, a_1), ..., Q(s_t, a_N)]
        # where N is the number of actions and s_t was the true action take at state s_t
        # it basically compresses the table of Q values to a colum vector only selecting the value from the
        # column corresponding to the action taken
        state_action_values = self.net(state_batch).gather(1, action_batch)

        # compute the targets r(s_t, a_t) + gamma max_{a} Q(s_{t+1}, a) if not done and r(s_t, a_t)
        # otherwise for each (s_t, a_t) in the batch
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.net(non_final_next_states).max(1).values

        expected_state_action_values = reward_batch + self.gamma * next_state_values
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optim.zero_grad()
        # clip the grad values for stability
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 100)
        self.optim.step()

    def train(self, episodes=100, samples=10, batch_size=10, print_every=10):
        env = self.env
        for episode in tqdm(range(1, episodes+1)):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            env.render()
            avg_reward = []
            for t in range(samples):
                action = self.act(state)
                next_state, reward, terminate, truncated, _ = env.step(action)
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0)
                if terminate:
                    next_state = None
                else:
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.buffer.push(state, action, next_state, reward)
                avg_reward.append(reward.item())
                state = next_state
                if self.buffer.is_full():
                    self.step(batch_size)
            avg_reward = torch.mean(torch.tensor(avg_reward)).item()
            if episode % print_every == 0:
                print("Episode: {:d}, Average Reward: {:.1f}".format(episode, avg_reward))

        self.env.close()








