#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:34:02 2020

@author: schroeti
"""
#to install package from GitHub
#1. copy github repo link
#2. open terminal
#3. type "git clone" and paste link
#4. ls
#5. cd to package name
#6. ls
#7. type "python setup.py develop"
#7b. to pull, go to gym_tpfood folder and type "git pull origin master"

# =============================================================================
# Import libraries
# =============================================================================

import gym
import gym_tpfood
import time
import numpy as np
import random
import pandas as pd
from IPython.display import clear_output
import matplotlib as plt
import datetime
import seaborn as sns
import latex
import math
import sys
from contextlib import closing
from io import StringIO
from gym import utils
from scipy.signal import savgol_filter
import optuna
import plotly
from plotly.offline import plot
import plotly.graph_objs as go
import socket


# =============================================================================
# Define graph parameters
# =============================================================================

## Plot graphic options
#to get Greek letters
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
print(os.getenv("PATH"))
import latex
import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import rcParams
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


from distutils.spawn import find_executable
if find_executable('latex'): print("latex installed")

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# =============================================================================
# Set path to save images
# =============================================================================

os.chdir(os.getcwd()+'/Desktop/RL/Plots_thesis/Q-learning')

# =============================================================================
# 1. Definition of environment
# =============================================================================

env = gym.make('Delivery-v0')

env.reset()
env.render()


action_size = env.action_space.n
print("There are", action_size, "actions.")

state_size = env.observation_space.n
print("The number of states is",state_size,".")


# =============================================================================
# 2. Definition of Q-learning function - Optuna - Hyperparameter tuning
# =============================================================================

def objective(trial): 
    """objective looks for an optimal set of hyperparameters for the Q-learning
        algorithm

    Args:
        trial: single execution of the objective function internally 
               instantiated upon each invocation of the function.


    Returns:
        Value of the last reward obtained in an episode
    """
    %time
    env = gym.make('Delivery-v0')
    alpha = trial.suggest_discrete_uniform('alpha', 0.3,0.9,0.3)
    gamma = trial.suggest_discrete_uniform('gamma', 0.6, 1,0.1)
    epsilon = trial.suggest_discrete_uniform('epsilon', 0.01, 0.11, 0.04)
    episodes = 1000000
    
    # For plotting metrics
    all_epochs = []
    all_penalties = []
    rewards = []
    
    #Initialize Q table of 22500 x 8 size (22500 states and 8 actions) with all zeroes
    q_table = np.zeros([env.observation_space.n, env.action_space.n])  
    
    for i in range(1, episodes+1):
        state = env.reset()
        episode_rewards = []

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space randomly
            else:
                action = np.argmax(q_table[state]) # Exploit learned values by choosing optimal values

            next_state, reward, done, info = env.step(action) 

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1
                

            state = next_state
            episode_rewards.append(reward)
            epochs += 1
        
            if done == True:
                  break 
            if epochs == 1000:
                break  
        rewards.append(np.sum(episode_rewards))
        
    last_reward = np.mean(rewards)
    # trial.report(-1 * last_reward)

    return -1 * last_reward
        
        

study = optuna.create_study()
study.optimize(objective, n_trials=30)

 
study.best_params

#Extract best values of parameters
best_alpha = study.best_params.get('alpha')
best_gamma = study.best_params.get('gamma')
best_epsilon = study.best_params.get('epsilon')

print(best_alpha)
print(best_gamma)
print(best_epsilon)

#Parameters importance
importances = optuna.importance.get_param_importances(study)

#Plot parameters relative importance with Plotly
imp = optuna.visualization.plot_param_importances(study)
plot(imp)

#Plot pptimization history with Plotly
opti = optuna.visualization.plot_optimization_history(study)
plot(opti)

#Export dataframe
trials_df_q_learning = study.trials_dataframe()
pd.DataFrame(trials_df_q_learning).to_csv("trials_df_q_learning.csv")

# =============================================================================
# 3. Train for a specific set of parameters and display training plot
# =============================================================================

best_alpha = 0.9
best_gamma = 0.8
best_epsilon= 0.01

def Q_learning_train(env,alpha,gamma,epsilon,episodes): 
    """Q_learning_train trains a Q-learning policy associated with input parameters

    Args:
        env: Environment 
        alpha (float): Learning Rate --> Extent to which our Q-values are being updated in every iteration.
        gamma (float): Discount Rate --> How much importance we want to give to future rewards
        epsilon (float): Probability of selecting random action instead of the 'optimal' action
        episodes (int): No. of episodes to train on

    Returns:
        Q-learning Trained policy
        A plot object of the rewards agains the number of episodes for the 
        training phase

    """
    %time
    # For plotting metrics
    all_epochs = []
    all_penalties = []
    rewards = []
    
    #Initialize Q table of 22500 x 8 size (22500 states and 8 actions) with all zeroes
    q_table = np.zeros([env.observation_space.n, env.action_space.n])  
    
    for i in range(1, episodes+1):
        state = env.reset()
        episode_rewards = []

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space randomly
            else:
                action = np.argmax(q_table[state]) # Exploit learned values by choosing optimal values

            next_state, reward, done, info = env.step(action) 

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1
                

            state = next_state
            episode_rewards.append(reward)
            epochs += 1
        
            if done == True:
                  break 
            if epochs == 1000:
                break  
        rewards.append(np.sum(episode_rewards))
            
        if i % 1000 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")
            
        
    print("Training finished.\n")
    
    plt.plot(savgol_filter(rewards, 1001, 3, mode = "interp"))
    plt.title("Smoothened training reward per episode", pad = 30, size = BIGGER_SIZE)
    plt.legend()
    plt.xlabel('Episodes', labelpad = 20);
    plt.ylabel('Total Reward', labelpad = 20);
    plt.tick_params(axis='both', which='major');
    plt.tick_params(axis='both', which='minor');
    #plt.xlim(0, 60000);
    #plt.ylim(0,50)
    #plt.xticks(np.arange(0, episodes+1, 5000));
    #plt.yticks(np.arange(min(rewards), max(rewards)+1, 1000));
    

plt.figure(figsize=(12,9)) 

#Best agent
Q_learning_train(env,best_alpha,best_gamma, best_epsilon, 200000)

#Worst agent
Q_learning_train(env,0.6,0.6,0.09,200000)

#Random agent
Q_learning_train(env,0.3,0.6,0.1,200000)

plt.legend([r'$\alpha$ = 0.9, $\gamma$ = 0.8, $\varepsilon$ = 0.01',
            r'$\alpha$ = 0.6, $\gamma$ = 0.6, $\varepsilon$ = 0.09',
            r'$\alpha$ = 0.3, $\gamma$ = 0.6, $\varepsilon$ = 0.1',
            ], loc = 'lower right')


plt.axhline(0, linestyle = "--", c = "black")
plt.savefig('q_learning_training_all1.png', dpi=600)

# =============================================================================
# 4. Evaluating Q-learning - Best results
# =============================================================================

def Q_learning_train_table(env,alpha,gamma,epsilon,episodes): 
    """Q_learning_train_table produces the Q-table associated with a trained Q-learning policy

    Args:
        env: Environment 
        alpha(float): Learning Rate --> Extent to which our Q-values are being updated in every iteration.
        gamma(float): Discount Rate --> How much importance we want to give to future rewards
        epsilon(float): Probability of selecting random action instead of the 'optimal' action
        episodes(int): No. of episodes to train on

    Returns:
        Q-table for a given set of Q-learning hyperparameters

    """
    %time
    # For plotting metrics
    all_epochs = []
    all_penalties = []
    rewards = []
    
    #Initialize Q table of 22500 x 8 size (22500 states and 8 actions) with all zeroes
    q_table = np.zeros([env.observation_space.n, env.action_space.n])  
    
    for i in range(1, episodes+1):
        state = env.reset()
        episode_rewards = []

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space randomly
            else:
                action = np.argmax(q_table[state]) # Exploit learned values by choosing optimal values

            next_state, reward, done, info = env.step(action) 

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1
                

            state = next_state
            episode_rewards.append(reward)
            epochs += 1
        
            if done == True:
                  break 
            if epochs == 1000:
                break  
        rewards.append(np.sum(episode_rewards))
            
        if i % 1000 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")
                
    print("Training finished.\n")
    
    return q_table

#Best agent Q-table
q_table_best = Q_learning_train_table(env, best_alpha, best_gamma, best_epsilon, 200000)

#Worst agent Q-table
q_table_worse = Q_learning_train_table(env, 0.6, 0.6, 0.06, 200000)

#Random agent Q-table
q_table_rand = Q_learning_train_table(env, 0.3, 0.6, 0.1, 200000)



def Q_learning_test(env,alpha,gamma,episodes, q_table): 
    """Q_learning_test tests a given Q-learning policy 

    Args:
        env: Environment 
        alpha(float): Learning Rate --> Extent to which our Q-values are being updated in every iteration.
        epsilon(float): Probability of selecting random action instead of the 'optimal' action
        episodes(int): No. of episodes to train on
        q_table

    Returns:
        Rewards for an evaluated Q-learning policy
        A plot object of the rewards agains the number of episodes for the 
        testing phase

    """
    %time
    
    # For plotting metrics
    all_epochs = []
    all_penalties = []
    rewards = []
    
    total_reward = 0
    
    for i in range(1, episodes+1):
        state = env.reset()
        episode_rewards = []

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            
            action = np.argmax(q_table[state]) # Exploit learned values by choosing optimal values
            next_state, reward, done, info = env.step(action) 


            if reward == -10:
                penalties += 1
                
            state = next_state
            episode_rewards.append(reward)
            epochs += 1
        
            if done == True:
                  break 
            if epochs == 1000:
                break  
            
            total_reward += reward
        rewards.append(np.sum(episode_rewards))
            
        if i % 1000 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

        
    print("Training finished.\n")
    
    
    plt.plot(savgol_filter(rewards, 1001, 3, mode = "interp"))
    plt.title("Smoothened testing reward per episode", pad = 30 , size = BIGGER_SIZE)
    plt.xlabel('Episodes', labelpad = 20);
    plt.ylabel('Total Reward', labelpad = 20);
    plt.tick_params(axis='both', which='major', labelsize=16);
    plt.tick_params(axis='both', which='minor', labelsize=16);
    #plt.xlim(100000, 200000);
    #plt.ylim(0,50)
    # plt.xticks(np.arange(0, episodes+1, 5000));
    # plt.yticks(np.arange(min(rewards), max(rewards)+1, 1000));
    

plt.figure(figsize=(12,9)) 


#Plot figures
Q_learning_test(env,best_alpha, best_gamma, 200000, q_table_best)
Q_learning_test(env,0.6,0.6,200000, q_table_worse)
Q_learning_test(env,0.3,0.6,200000, q_table_rand)
#Worst agent


plt.legend([r'$\alpha$ = 0.9, $\gamma$ = 0.8, $\varepsilon$ = 0.01',
            r'$\alpha$ = 0.6, $\gamma$ = 0.6, $\varepsilon$ = 0.09',
            r'$\alpha$ = 0.3, $\gamma$ = 0.6, $\varepsilon$ = 0.1',
            ], loc = 'lower right')

plt.axhline(0, linestyle = "--", c = "black")
plt.savefig('q_learning_testing.png', dpi=1200)



# =============================================================================
# 5. Creating frames for video for best agent
# =============================================================================

total_epochs, total_penalties, total_reward = 0, 0, 0
episodes = 10000
frames = []
got_right = 0
nb_steps = 0
d = {}

#Takes trained Q-table of best agent (cf. section 4)
q_table = q_table_best

for ep in range(episodes):
    total_reward_by_ep = 0
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    steps = 0

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        new_action = np.argmax(q_table[state])
        
        if reward == -10:
            penalties += 1
        
        if epochs == 1000:
            break  

        total_reward_by_ep += reward
        total_reward += reward
        
        steps += 1
        nb_steps += 1
        total_penalties += penalties

        #Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'episode': ep, 
            'state': state,
            'action': action,
            'new action': new_action,
            'reward': reward, 
            'total reward': total_reward_by_ep
            }
        )
        


print(f"Results after {episodes} episodes:")
print(f"Total number of steps : {nb_steps}")
print(f"Average timesteps per episode: {np.sum(nb_steps)/episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
print(f"Average reward per episode: {total_reward / episodes}")
print(f"Average reward per timestep: {total_reward / nb_steps}")



# =============================================================================
# 6. Export Q-table of best combination
# =============================================================================
#Set path to save q_table
os.chdir(os.getcwd()+'/Desktop/RL/Plots_thesis/Q-learning')


#Define function to highlight best action by state
def highlight_max(s):    
    is_max = s == s.max()
    return ['background-color: orange' if v else '' for v in is_max]

#Export array as csv file
df = pd.DataFrame(q_table_best).style.apply(highlight_max, axis = 1)

#Export array as csv file
pd.DataFrame(q_table_best).to_csv("q_table_best_q_learning.csv", index_label = "state", header =["South", "North", "East", "West", "Pickup1", "Dropoff1", "Pickup2", "Dropoff2"],float_format='%.5f' )

#Export array as xls file
df.to_excel("q_table_best_q_learning.xlsx",index_label = "state", header =["South", "North", "East", "West", "Pickup1", "Dropoff1","Pickup2", "Dropoff2" ],float_format='%.5f' )


# =============================================================================
# #7.  Define function to print given episode
# =============================================================================
def print_frames(frames):
    """print_frames produces a visual sequence of the environment for one or
       several episodes

    Args:
        frames (list): an array of hidden environment sequences in one or several
        episodes

    Returns:
        A visual sequence of printed environment sequences for one or several 
        episodes

    """
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Episode: {frame['episode']}")
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Previous action: {frame['action']}")
        if frame['action'] == 0:
            print("Action is: south")
        if frame['action'] == 1:
            print("Action is: north")
        if frame['action'] == 2:
            print("Action is: east")
        if frame['action'] == 3:
            print("Action is: west")
        if frame['action'] == 4:
            print("Action is: pickup passenger 1 ") 
        if frame['action'] == 5:
            print("Action is: dropoff passenger 1")
        if frame['action'] == 6:
            print("Action is: pickup passenger 2")
        if frame['action'] == 7:
            print("Action is: dropoff passenger 2")
        print(f"Reward: {frame['reward']}")
        print(f"Total Reward: {frame['total reward']}")
        time.sleep(.5)
        
# =============================================================================
# 8. Print best agent episodes
# =============================================================================        
# print episodes
print_frames(frames_eval)


# =============================================================================
# 9. Rendering of special cases
# =============================================================================
# Some states such as state 12, 19094 or 22125 are promising cases of 
# ride-sharing. They were identified manually in the Q-learning algorithm part.
# To determine whether the agent uses the opportunity to pick up
# both passengers at the same time, we decide to print the episode starting
# from state 12, 19094 or 22125 until the end. 


# Reset env
state = env.reset()
# Present trained behaviour over episodes
num_test_episodes = 1

episodes_passed = 0
acc_episodic_reward = 0.0
episode_reward = []
frames = []

env.s = 12 #19094, 22125

while episodes_passed < num_test_episodes:
    # Choose action greedily
    action = trained_AC_model.select_action(state)
    # Act on env
    state, reward, done, _ = env.step(action)
    # Add to accumulative reward
    acc_episodic_reward += reward

    frames.append({
          'frame': env.render(mode='ansi'),
          'episode':'state 12',
          'state': state,
          'action': action,
          'reward': reward
          }
      )
  
    # When episode is done - reset and print (limit to 200 transitions in test)
    if done:
        # Print acc. reward
        print('Episode {}\tAccumulated Reward: {:.2f}\t'.format(
            episodes_passed+1, acc_episodic_reward))
        # Update statistics
        episodes_passed += 1
        episode_reward.append(acc_episodic_reward)
        acc_episodic_reward = 0.0

        state = env.reset()

np.mean(episode_reward)
        
# print episode
print_frames(frames)
 