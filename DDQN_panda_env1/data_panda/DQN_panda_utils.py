import gym
import torch.nn as nn
import torch
import numpy as np
from scipy.signal import convolve, gaussian
import os
import io
import base64
from IPython.display import HTML


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQNAgent(nn.Module):
    def __init__(self, state_shape, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.actions_space = self.create_actions_space()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.state_shape = state_shape
        state_dim = state_shape
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        self.network = nn.Sequential()
        self.network.add_module('layer1', nn.Linear(state_dim, 128))
        self.network.add_module('relu1', nn.ReLU())
        self.network.add_module('layer2', nn.Linear(128, 256))
        self.network.add_module('relu2', nn.ReLU())
        self.network.add_module('layer4', nn.Linear(256, self.n_actions))
        self.parameters = self.network.parameters

        self.get_action_index = lambda a: np.where(
            (self.actions_space == a).all(axis=1))[0][0]

    def forward(self, state_t):
        # pass the state at time t through the newrok to get Q(s,a)
        qvalues = self.network(state_t)
        return qvalues

    def create_actions_space(self):
        actions = [-1, 0, 1]
        actions_space = []
        # Create space of actions

        for j1 in actions:
            for j2 in actions:
                for j3 in actions:
                    actions_space.append([j1, j2, j3])
        actions_space.pop(13)  # remove [0,0,0]
        self.n_actions = len(actions_space)
        return actions_space

    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = np.array(states)
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        random_actions = self.actions_space[random_actions[0]]
        best_actions = qvalues.argmax(axis=-1)
        best_actions = self.actions_space[best_actions[0]]
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

    def play(self, env, NAME_DIR, tmax=300, model="best", q0=None):
        
        if model == "last":
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/last-model.pt'))
        else:

            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/best-model.pt'))
        if len(q0)!=7:
            s = env.reset()
        else:
            env.panda.q=q0
            s=env.get_state()
        reward = 0
        for step in range(tmax):
            qvalues = self.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0]
            s, r, done, info = env.step(self.actions_space[action])
            reward += r
            if done or info[0] == "terminated":
                break
        print(f'Final score:{reward} in {step} steps')
        print(f"Status: {info[0]} {info[1]}")


def evaluate(env, agent, n_games=1, greedy=False, t_max=100):
    rewards = []
    steps = []
    infos = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for step in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(
                axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, info = env.step(agent.actions_space[action])
            reward += r
            # print(reward)
            if done or info[0] == "terminated":
                #print(f"Done with Steps: {steps}")
                # print(info)

                break
        infos.append(info)
        steps.append(step)
        rewards.append(reward)
    return np.mean(rewards), np.mean(steps), infos


def compute_td_loss(agent, target_network, states, actions, rewards, next_states, done_flags,
                    gamma=0.99, device=device):

    # convert numpy array to torch tensors
    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    done_flags = torch.tensor(done_flags.astype(
        'float32'), device=device, dtype=torch.float)

    # get q-values for all actions in current states
    # use agent network
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    # use target network
    predicted_next_qvalues = target_network(next_states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), actions]

    # compute Qmax(next_states, actions) using predicted next q-values
    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    # compute "target q-values"
    target_qvalues_for_actions = rewards + \
        gamma * next_state_values * (1-done_flags)

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)

    return loss


class ReplayBuffer:
    def __init__(self, size):
        self.size = size  # max number of items in buffer
        self.buffer = []  # array to holde buffer
        self.next_id = 0

    # overloading len method to be linked to len of buffer
    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)


def play_and_record(start_state, agent, env, exp_replay, n_steps=1):

    s = start_state
    sum_rewards = 0

    # Play the game for n_steps and record transitions in buffer
    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])
        a = agent.sample_actions(qvalues)
        next_s, r, done, info = env.step(a)
        # print(r)
        sum_rewards += r
        exp_replay.add(s, a, r, next_s, done)
        if done or info[0] == "terminated":
            s = env.reset()
        else:
            s = next_s

    return sum_rewards, s


def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps-start_eps)*min(step, final_step)/final_step


def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, 'valid')


def generate_animation(env, agent, save_dir):

    try:
        env = gym.wrappers.Monitor(
            env, save_dir, video_callable=lambda id: True, force=True, mode='evaluation')
    except gym.error.Error as e:
        print(e)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state = env.reset()
    reward = 0
    steps = 0
    while True:
        qvalues = agent.get_qvalues([state])
        action = qvalues.argmax(axis=-1)[0]
        state, r, done, _ = env.step(action)
        steps += 1
        reward += r
        if done or (steps == 500):
            print('Got reward: {}'.format(reward))
            break


def display_animation(filepath):
    video = io.open(filepath, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))


def compute_td_loss_priority_replay(agent, target_network, replay_buffer,
                                    states, actions, rewards, next_states, done_flags, weights, buffer_idxs,
                                    gamma=0.99, device=device):

    # convert numpy array to torch tensors
    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    done_flags = torch.tensor(done_flags.astype(
        'float32'), device=device, dtype=torch.float)
    weights = torch.tensor(weights, device=device, dtype=torch.float)

    # get q-values for all actions in current states
    # use agent network
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    # use target network
    predicted_next_qvalues = target_network(next_states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), actions]

    # compute Qmax(next_states, actions) using predicted next q-values
    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    # compute "target q-values"
    target_qvalues_for_actions = rewards + \
        gamma * next_state_values * (1-done_flags)

    # compute each sample TD error
    loss = ((predicted_qvalues_for_actions -
            target_qvalues_for_actions.detach()) ** 2) * weights

    # mean squared error loss to minimize
    loss = loss.mean()

    # calculate new priorities and update buffer
    with torch.no_grad():
        new_priorities = predicted_qvalues_for_actions.detach() - \
            target_qvalues_for_actions.detach()
        new_priorities = np.absolute(new_priorities.detach().cpu().numpy())
        replay_buffer.update_priorities(buffer_idxs, new_priorities)

    return loss
