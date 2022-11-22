import torch
import json
import pickle
import yaml
import random
import torch.nn as nn
import numpy as np
from tqdm import trange
import data_panda as rbt
from torch.utils.tensorboard import SummaryWriter
import glob
import os

# Reinforcement with reward to go

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
if device == "cpu":
    print("gpu is not available")
    print(device)
    exit()

# folder to load config file

# Function to load yaml configuration file


def load_config(config_name):
    CONFIG_PATH = "./config/"

    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = rbt.load_config("config_REINF.yaml")

state_shape = config["state_shape"]
# Environment settings
env = rbt.Panda_RL()
env.renderize = config["renderize"]  # stop robot viewing
env.delta = config["delta"]
env.fg = config["fitness"]  # fitness goal
env.ceil = config['ceil']
env.bonus_complete = config['success_rw']

agent = rbt.REINF(state_shape, device, epsilon=1).to(device)


RESTORE_AGENT = config["RESTORE_AGENT"]  # Restore a trained agent

env.step_penalty = config["step_penalty"]
env.collision_penalty = config["collision_penalty"]
env.sig_p = config["sig_p"]
env.sig_R = config["sig_R"]


# set a seed
seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


RES_DIR = rbt.set_res_dir(comment=config["comment"])
comment = ""
# monitor_tensorboard()
tb = SummaryWriter(log_dir=RES_DIR, comment=comment)

LOAD_MODEL = config["LOAD_MODEL"]
if LOAD_MODEL:
    folder = config["resume_folder"]
    agent.load_weights(folder, model=config["model_resume"])
    percentage_of_total_steps = config["percentage_of_total_steps_resume"]
    print(f"Loaded {folder} {config['model_resume']} ")
    # print(f"Restored  {folder}")


tmax = config["tmax"]

env.mag = config["mag"]


# monitor_tensorboard()
tb = SummaryWriter(log_dir=RES_DIR, comment=comment)

# setup some parameters for training


batch_size = config["batch_size"]
total_steps = config["total_steps"]
# total_steps = 10

# init Optimizer
lr = 10 ** config["lr_exp"]
opt = torch.optim.Adam(agent.parameters(), lr=lr)


# setup some frequency for logging and updating target network
eval_freq = config["eval_freq"]

# to clip the gradients
max_grad_norm = config["max_grad_norm"]


hyperparameters_train = {
    "lr": lr,
    "batch_size": batch_size,
    "total_steps": total_steps,
    "tmax": tmax
    # "agent": str(agent.network)
}


def save_hyperparameter(dict, directory):
    with open(directory + "/hyperparameters.json", "w") as outfile:
        json.dump(dict, outfile)


# Start training
# init Optimizer
optimizer = torch.optim.Adam(agent.network.parameters(), lr=lr)

states = env.reset()


def train_one_episode(states, actions, rewards, gamma=0.99, entropy_coef=1e-2):

    # get rewards to go
    rewards_to_go = agent.get_rewards_to_go(rewards, gamma)

    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards_to_go = torch.tensor(
        rewards_to_go, device=device, dtype=torch.float)

    # get action probabilities from states
    logits = agent.network(states)
    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)

    log_probs_for_actions = log_probs[range(len(actions)), actions]

    # Compute loss to be minized
    J = torch.mean(log_probs_for_actions*rewards_to_go)
    H = -(probs*log_probs).sum(-1).mean()

    loss = -(J+entropy_coef*H)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return np.sum(rewards)  # to show progress on training


total_rewards = []
rw_min = -np.inf
print(f"Device: {device}")
print(f"Ceil: {env.ceil}")
success = 0
collisions = 0
rw=[]

for i in trange(config["total_steps"]):

    
    states, actions, rewards, info = agent.generate_trajectory(
        env, n_steps=config['tmax'])
    reward = train_one_episode(states, actions, rewards)
    #total_rewards.append(reward)
    rw.append(np.sum(rewards))

    if info[1] == "Completed":
        success += 1
    elif info[1] == "Collided":
        collisions += 1

    if i % config["eval_freq"] == 0:
        mean_reward = np.mean(rw)
        tb.add_scalar("1/reward", mean_reward,i)
        tb.add_scalar("1/success", success/config["eval_freq"], i)
        tb.add_scalar("1/collisions", collisions/config["eval_freq"], i)
        #reset
        rw=[]
        
        success = 0
        collisions = 0
        torch.save(agent.state_dict(), RES_DIR + "/last-model.pt")
        if mean_reward > rw_min:
            torch.save(agent.state_dict(), RES_DIR + "/best-model-rw.pt")
            rw_min = mean_reward
        print("mean reward:%.3f" % (mean_reward))
        # if mean_reward > 300:
        #     break
tb.close()


# state = env.reset()
# # tb.add_graph(agent.network, torch.tensor(
# #     state, device=device, dtype=torch.float32))
# save_hyperparameter(hyperparameters_train, RES_DIR)
# loss_min = np.inf
# rw_min = -np.inf
# print(f"buffer size = {len(exp_replay)} ")
# print(f"Frequency evaluation = {eval_freq}")
# print(f"Device: {device}")


# for step in trange(total_steps + 1, desc="Training", ncols=70):

#     # reduce exploration with learning progress
#     agent.epsilon = rbt.epsilon_schedule(
#         start_epsilon, end_epsilon, step, eps_decay_final_step
#     )

#     # take timesteps_per_epoch and update experience replay buffer
#     _, state = rbt.play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

#     # train by sampling batch_size of data from experience replay
#     (
#         states,
#         actions,
#         rewards,
#         next_states,
#         done_flags,
#         weights,
#         idxs,
#     ) = exp_replay.sample(batch_size)
#     actions = [agent.get_action_index(i) for i in actions]

#     # loss = <compute TD loss>
#     opt.zero_grad()
#     loss = rbt.compute_td_loss_priority_replay(
#         agent,
#         target_network,
#         exp_replay,
#         states,
#         actions,
#         rewards,
#         next_states,
#         done_flags,
#         weights,
#         idxs,
#         gamma=0.99,
#         device=device,
#     )
#     loss.backward()
#     grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
#     opt.step()
#     if loss < loss_min:
#         torch.save(agent.state_dict(), RES_DIR + "/best-model-loss.pt")
#         loss_min = loss
#     tb.add_scalar("2/Epsilon", agent.epsilon, step)
#     tb.add_scalar("1/TD Loss", loss, step)

#     if step % refresh_target_network_freq == 0:
#         # Load agent weights into target_network
#         target_network.load_state_dict(agent.state_dict())

#     if step % eval_freq == 0:
#         # eval the agent
#         assert not np.isnan(loss.cpu().detach().numpy())
#         # clear_output(True)
#         m_reward, m_steps, m_collisions, m_successes, fit, _ = rbt.evaluate(
#             env, agent, n_games=config["n_games_eval"], greedy=True, t_max=tmax
#         )
#         tb.add_scalar("1/Rw", m_reward, step)
#         tb.add_scalar("1/steps", m_steps, step)
#         tb.add_scalar("2/fitness reached", fit, step)
#         tb.add_scalar("2/Collisions", m_collisions, step)
#         tb.add_scalar("1/Successes", m_successes, step)
#         # print(f"Last mean reward = {m_reward}")

#     if m_reward > rw_min:
#         torch.save(agent.state_dict(), RES_DIR + "/best-model-rw.pt")
#         rw_min = m_reward

#     # clear_output(True)
# exp_replay.save_buffer(RES_DIR)
# torch.save(agent.state_dict(), RES_DIR + "/last-model.pt")
# tb.close()
# rbt.eval_trained_models(env, agent, RES_DIR, device)

# env.renderize=True
# q_far=np.array([ 0., -0.8 ,  0. , -0.0698,  0.,  3.3825,  0.    ])
# NAME_DIR=RES_DIR.split("/")[1]

# agent.play(env,NAME_DIR,tmax=800)
