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
import joblib
import os
from optuna.trial import TrialState
import optuna
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device=='cpu':
    print('gpu is not available')
    print(device)
    exit()

# folder to load config file




config = rbt.load_config("config_dueling.yaml")

def objective(trial):
    state_shape = config['state_shape']
    env = rbt.Panda_RL()
    env.renderize = config['renderize']  # stop robot viewing
    env.delta = config['delta']
    agent = rbt.DQNAgent(state_shape, device, epsilon=1).to(device)

    #init agent, target network and Optimizer
    agent = rbt.DuelingDQNAgent(state_shape,device,config["dueling_layers"], epsilon=1).to(device)
    target_network = rbt.DuelingDQNAgent(state_shape,device, config["dueling_layers"], epsilon=1).to(device)
    target_network.load_state_dict(agent.state_dict())
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)


    RESTORE_AGENT = config['RESTORE_AGENT']  # Restore a trained agent
    NEW_BUFFER = config['NEW_BUFFER']   # Restore a buffer
    TRAIN = config['TRAIN']  # Train or only simulate
    
    

    # agent.n_layer1=config['layer1']
    # agent.n_layer2=config['layer1']*2

    env.step_penalty=config['step_penalty']
    env.collision_penalty=config['collision_penalty']
    env.sig_p=config['sig_p']
    env.sig_R=config['sig_R']


    # set a seed
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Fill buffer with samples collected ramdomly from environment
    buffer_len = trial.suggest_int('Buffer len',1000,20000)
     
    exp_replay = rbt.PrioritizedReplayBuffer(buffer_len)
    RES_DIR = rbt.set_res_dir(comment=config['comment'])
    comment = ""
    # monitor_tensorboard()
    tb = SummaryWriter(log_dir=RES_DIR, comment=comment)

    LOAD_MODEL=config['LOAD_MODEL']
    if LOAD_MODEL:
        folder=config['resume_folder']
        agent.load_weights(folder,model=config['model_resume'])
        percentage_of_total_steps=config['percentage_of_total_steps_resume']
        print(f"Loaded {folder} {config['model_resume']} ")
        #print(f"Restored  {folder}")  


    if NEW_BUFFER:
        for i in trange(500,desc="Buffering",ncols=70):

            state = env.reset()
            # Play 100 runs of experience with 100 steps and  stop if reach 10**4 samples
            rbt.play_and_record(state, agent, env, exp_replay, n_steps=50)

            if len(exp_replay) == buffer_len:
                break
        print(f"New buffer with {len(exp_replay)} samples")
    else:
        exp_replay = rbt.PrioritizedReplayBuffer(buffer_len)
        exp_replay.load_buffer(config['resume_folder'])
        



    tmax = config['tmax']
    env.reset_j1=config['reset_j1']
    env.mag=config['mag']




    # monitor_tensorboard()
    tb = SummaryWriter(log_dir=RES_DIR, comment=comment)

    percentage_of_total_steps = config['percentage_of_total_steps']



    # setup some parameters for training

    timesteps_per_epoch = config['timesteps_per_epoch']
    timesteps_per_epoch = trial.suggest_int('timesteps_per_epoch',1,3)
    batch_size = config['batch_size']
    total_steps = config['total_steps']
    #total_steps = 10

    # init Optimizer
    lr = config['lr']
    lr=trial.suggest_int('lr',0.0001,0.001)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)

    # set exploration epsilon
    start_epsilon = config['start_epsilon']
    #start_epsilon = 0.1
    end_epsilon = config['end_epsilon']
    eps_decay_final_step = percentage_of_total_steps*total_steps

    # setup some frequency for logging and updating target network
    loss_freq = config['loss_freq']
    refresh_target_network_freq = trial.suggest_int('Refresh rate',5,200)
    eval_freq = config['eval_freq']

    # to clip the gradients
    max_grad_norm = config['max_grad_norm']




    hyperparameters_train = {"start_epsilon": start_epsilon,
                            "end_epsilon": end_epsilon,
                            "lr": lr,
                            "batch_size": batch_size,
                            "total_steps": total_steps,
                            "percentage_of_total_steps": percentage_of_total_steps,
                            "refresh_target_network_freq": refresh_target_network_freq,
                            "buffer_len": buffer_len,
                            "tmax": tmax
                            #"agent": str(agent.network)
                            }


    def save_hyperparameter(dict, directory):
        with open(directory+"/hyperparameters.json", "w") as outfile:
            json.dump(dict, outfile)


    # Start training
    state = env.reset()
    # tb.add_graph(agent.network, torch.tensor(
    #     state, device=device, dtype=torch.float32))
    save_hyperparameter(hyperparameters_train, RES_DIR)
    loss_min = np.inf
    rw_max=-np.inf
    print(f"buffer size = {len(exp_replay)} ")  
    print(f"Frequency evaluation = {eval_freq}")
    print(f"Device: {device}")



    for step in trange(total_steps + 1, desc="Training", ncols=70):


        # reduce exploration as we progress
        agent.epsilon = rbt.epsilon_schedule(
            start_epsilon, end_epsilon, step, eps_decay_final_step)

        # take timesteps_per_epoch and update experience replay buffer
        _, state = rbt.play_and_record(
            state, agent, env, exp_replay, timesteps_per_epoch)

        # train by sampling batch_size of data from experience replay
        states, actions, rewards, next_states, done_flags, weights, idxs = exp_replay.sample(
            batch_size)
        actions = [agent.get_action_index(i) for i in actions]

        # loss = <compute TD loss>
        opt.zero_grad()
        loss = rbt.compute_td_loss_priority_replay(agent, target_network, exp_replay,
                                                states, actions, rewards, next_states, done_flags, weights, idxs,
                                                gamma=0.99,
                                                device=device)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        opt.step()
        if loss < loss_min:
            torch.save(agent.state_dict(), RES_DIR+'/best-model-loss.pt')
            loss_min=loss
        tb.add_scalar("1/Epsilon", agent.epsilon, step)
        tb.add_scalar("1/TD Loss", loss, step)

        if step % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())

        if step % eval_freq == 0:
            # eval the agent
            assert not np.isnan(loss.cpu().detach().numpy())
            #clear_output(True)        
            m_reward,m_steps,m_collisions,m_successes,fit,_ = rbt.evaluate(env, agent, n_games=10,
                                    greedy=True, t_max=tmax)
            tb.add_scalar("1/Mean reward per episode", m_reward, step)
            tb.add_scalar("1/Mean of steps", m_steps, step)
            tb.add_scalar("2/Mean fitness reached", fit, step)
            tb.add_scalar("2/Mean of collisions", m_collisions, step)
            tb.add_scalar("2/Mean of successes", m_successes, step)
            #print(f"Last mean reward = {m_reward}")
 
        if m_reward > rw_max:
            torch.save(agent.state_dict(), RES_DIR+'/best-model-rw.pt')
            rw_max=m_reward
            
        
        #clear_output(True)
    exp_replay.save_buffer(RES_DIR)
    torch.save(agent.state_dict(), RES_DIR+'/last-model.pt')
    tb.close()
    rbt.eval_trained_models(env,agent,RES_DIR,device)
    
    return rw_max

# env.renderize=True
# q_far=np.array([ 0., -0.8 ,  0. , -0.0698,  0.,  3.3825,  0.    ])
# NAME_DIR=RES_DIR.split("/")[1]

# agent.play(env,NAME_DIR,tmax=800)
    
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
joblib.dump(study, "study_new_env1.pkl")

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
