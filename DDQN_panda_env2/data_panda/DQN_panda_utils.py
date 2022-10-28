import gym
import torch
import numpy as np
from scipy.signal import convolve, gaussian
import os
import yaml
import io
import base64
import pickle
import json
import glob
from IPython.display import HTML

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load yaml configuration file
def load_config(config_name):
    CONFIG_PATH = "./config/"
    CONFIG_PATH
    
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def set_res_dir(TRAIN=True,comment=''):
    # Directory to store results
    res_dir_count = len(glob.glob('runs/*'))
    print(f"Current number of result directories: {res_dir_count}")
    if TRAIN:
        RES_DIR = "runs/"+comment+f"_{res_dir_count+1}"
        print(RES_DIR)
    else:
        RES_DIR = f"runs/results_{res_dir_count}"
    return RES_DIR

def eval_trained_models(env,agent,DIR,device):
    files=glob.glob(DIR+'/*.pt')
    
    for model in files:
        agent.load_state_dict(torch.load(model,map_location=device))
        print(model)
        
        m_reward,m_steps,m_collisions,m_successes,fit,_=evaluate(env,agent,n_games=20, greedy=True, t_max=500)
        print(m_reward,m_steps,m_collisions,m_successes,fit)
    print("m_reward,m_steps,m_collisions,m_successes,fit")
    
    


def evaluate(env, agent, n_games=1, greedy=False, t_max=100):
    rewards = []
    steps = []
    infos = []
    collisions=[]
    successes=[]
    fitness=[]
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        n_collisions=0
        success=0
        fit=100
        for step in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(
                axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, info = env.step(agent.actions_space[action])
            reward += r
            if np.sum(env.fitness())<fit:
                fit=np.sum(env.fitness())
            # print(reward)
            if info[1]=="Collided":
                n_collisions+=1
            if info[1]=="Completed":
                success+=1
            if done or info[0] == "Termination":
                #print(f"Done with Steps: {steps}")
                # print(info)
                break

        collisions.append(n_collisions)
        infos.append(info)
        steps.append(step)
        rewards.append(reward)
        successes.append(success)
        fitness.append(fit)
    return np.mean(rewards), np.mean(steps), np.mean(collisions), np.mean(successes), np.mean(fitness), infos


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
    
    def load_buffer(self,folder):
        with open('runs/'+folder+'/buffer.pickle', 'rb') as handle:
            self.buffer=pickle.load(handle)
        print("Replay Buffer Loaded")
        
    def save_buffer(self,folder):
        with open('runs/'+folder+'/buffer.pickle', 'wb') as handle:
            pickle.dump(self.buffer, handle)
        print("Replay Buffer Saved")


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


def td_loss_dqn(agent, target_network, states, actions, rewards, next_states, done_flags,
                    gamma=0.99, device=device):

    # convert numpy array to torch tensors
    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    done_flags = torch.tensor(done_flags.astype('float32'),device=device,dtype=torch.float)

    # get q-values for all actions in current states
    # use agent network
    q_s = agent(states)

    # select q-values for chosen actions
    q_s_a = q_s[range(
        len(actions)), actions]

    with torch.no_grad():
        # compute q-values for all actions in next states
        # use target network
        q_s1 = target_network(next_states)


        # compute Qmax(next_states, actions) using predicted next q-values
        q_s1_a1max,_ = torch.max(q_s1, dim=1)

        # compute "target q-values" 
        target_q = rewards + gamma * q_s1_a1max * (1-done_flags)

    # mean squared error loss to minimize
    loss = torch.mean((q_s_a - target_q.detach()) ** 2)

    return loss
