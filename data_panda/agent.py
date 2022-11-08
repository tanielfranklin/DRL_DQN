import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


class DQNAgent(nn.Module):
    def __init__(self, state_shape, device, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_layer1 = 128
        self.n_layer2 = 256
        self.actions_space = self.create_actions_space()
        self.device = device

        self.state_shape = state_shape
        state_dim = state_shape
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        self.network = nn.Sequential()
        self.network.add_module('layer1', nn.Linear(state_dim, self.n_layer1))
        self.network.add_module('relu1', nn.ReLU())
        self.network.add_module(
            'layer2', nn.Linear(self.n_layer1, self.n_layer2))
        self.network.add_module('relu2', nn.ReLU())
        self.network.add_module('layer4', nn.Linear(
            self.n_layer2, self.n_actions))
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

    def load_weights(self, NAME_DIR, model="best"):

        if model == "last":
            self.load_state_dict(torch.load(
                'runs/'+NAME_DIR+'/last-model.pt', map_location=self.device))
        elif model == "best":
            self.load_state_dict(torch.load(
                'runs/'+NAME_DIR+'/best-model-rw.pt', map_location=self.device))
        else:
            self.load_state_dict(torch.load(
                'runs/'+NAME_DIR+'/best-model-loss.pt', map_location=self.device))

    def play(self, env, NAME_DIR, tmax=500, model="best", q0=[], plot=False):

        if model == "last":
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/last-model.pt'))
        elif model == "best":
            self.load_state_dict(torch.load(
                'runs/'+NAME_DIR+'/best-model-rw.pt'))
        else:
            self.load_state_dict(torch.load(
                'runs/'+NAME_DIR+'/best-model-loss.pt'))

        if len(q0) != 7:
            s = env.reset()
        else:
            env.panda.q = q0
            s = env.get_state()

        dist = 100
        rewards = []
        distance = []
        for step in range(tmax+1):
            qvalues = self.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0]
            s, r, done, info = env.step(self.actions_space[action])
            rewards.append(r)
            distance.append(env.distance())

            if env.distance() < dist:
                dist = distance[-1]

            if done or info[1] == "Collided":
                break
        if info[0] == "Running" and info[0] == "":
            info[1] = "Truncated"

        print(
            f'Final score:{np.sum(rewards)} in {step} steps, minimum distance {dist}')
        print(f"Status: {info[0]} {info[1]}")
        if plot:
            cum_r = []
            j = 0
            for i in range(0, len(rewards)):
                j += rewards[i]
                cum_r.append(j)
            _, axis = plt.subplots(3, 1)
            axis[0].plot(rewards, label='rewards')
            axis[1].plot(cum_r, label='cumulative rewards')
            axis[2].plot(distance, label='distance')
            plt.legend()


class DuelingDQNAgent(nn.Module):
    def __init__(self, state_shape, device, layers=[64,256,64], epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.state_shape = state_shape
        self.actions_space = self.create_actions_space()
        self.device = device
        state_dim = state_shape
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        
        self.fc1 = nn.Linear(state_dim, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc_value = nn.Linear(layers[1], layers[2])
        self.fc_adv = nn.Linear(layers[1], layers[2])
        self.value = nn.Linear(layers[2], 1)
        self.adv = nn.Linear(layers[2], self.n_actions)
        #self.parameters = self.network.parameters

        self.get_action_index = lambda a: np.where(
            (self.actions_space == a).all(axis=1))[0][0]

    def forward(self, state_t):
        # pass the state at time t through the newrok to get Q(s,a)
        x = F.relu(self.fc1(state_t))
        x = F.relu(self.fc2(x))
        v = F.relu(self.fc_value(x))
        v = self.value(v)
        adv = F.relu(self.fc_adv(x))
        adv = self.adv(adv)
        adv_avg = torch.mean(adv, dim=1, keepdim=True)
        qvalues = v + adv - adv_avg
        return qvalues

    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = torch.tensor(
            np.array(states), device=self.device, dtype=torch.float32)
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

    def load_weights(self, NAME_DIR, model="best"):

        if model == "last":
            self.load_state_dict(torch.load(
                'runs/'+NAME_DIR+'/last-model.pt', map_location=self.device))
        elif model == "best":
            self.load_state_dict(torch.load(
                'runs/'+NAME_DIR+'/best-model-rw.pt', map_location=self.device))
        else:
            self.load_state_dict(torch.load(
                'runs/'+NAME_DIR+'/best-model-loss.pt', map_location=self.device))

    def play(self, env, NAME_DIR, tmax=500, model="best", q0=[], plot=False):

        if model == "last":
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/last-model.pt'))
        elif model == "best":
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/best-model-rw.pt'))
        else:
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/best-model-loss.pt'))

        if len(q0) != 7:
            s = env.reset()
        else:
            env.panda.q = q0
            s = env.get_state()

        dist = 100
        rewards = []
        distance = []
        for step in range(tmax+1):
            qvalues = self.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0]
            s, r, done, info = env.step(self.actions_space[action])
            rewards.append(r)
            distance.append(sum(env.fitness()))

            if sum(env.fitness()) < dist:
                dist = distance[-1]

            if done or info[1] == "Collided":
                break
        if info[0] == "Running" and info[0] == "":
            info[1] = "Truncated"

        print(
            f'Final score:{np.sum(rewards)} in {step} steps, minimum distance {dist}')
        print(f"Status: {info[0]} {info[1]}")
        if plot:
            cum_r = []
            j = 0
            for i in range(0, len(rewards)):
                j += rewards[i]
                cum_r.append(j)
            _, axis = plt.subplots(3, 1)
            axis[0].plot(rewards, label='rewards')
            axis[1].plot(cum_r, label='cumulative rewards')
            axis[2].plot(distance, label='distance')
            plt.legend()




class CategoricalDQN(nn.Module):
    def __init__(self, state_shape, n_actions, device, n_atoms=51, Vmin=-10, Vmax=10, epsilon=0):
        
        super(CategoricalDQN, self).__init__()
        self.epsilon = epsilon
        self.device=device
        self.n_actions = n_actions
        self.state_shape = state_shape        
        state_dim = state_shape
        self.n_atoms = n_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 32)
        self.probs = nn.Linear(32, n_actions * n_atoms)
        
    def forward(self, state_t):
        # pass the state at time t through the newrok to get Q(s,a)
        x = F.relu(self.fc1(state_t))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        probs = F.softmax(self.probs(x).view(-1, self.n_actions, self.n_atoms), dim=-1)
        return probs

    def get_probs(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        probs = self.forward(states)
        return probs.data.cpu().numpy()

    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        probs = self.forward(states)
        support = torch.linspace(self.Vmin, self.Vmax, self.n_atoms,device=self.device)
        qvals = support * probs
        qvals = qvals.sum(-1)
        return qvals.data.cpu().numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)