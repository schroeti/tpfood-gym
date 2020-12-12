#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:48:54 2020

@author: schroeti
"""


# Imports
import gym
import gym_tpfood
import numpy as np
from itertools import count

import torch
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import time
from IPython.display import clear_output
import matplotlib as plt


import random

from collections import namedtuple
import matplotlib.pyplot as plt
from torch.distributions import Categorical


# =============================================================================
# Define graph parameters
# =============================================================================
import latex
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
plt.rcParams.update({'font.size': 12})

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams['agg.path.chunksize'] = 10000000


RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


'''
    This script includes the implementations of all the necessary/auxiliary Classes/Functions:
    - AC_model - deep network for actor critic learning
    - unpack_arch - unpacking input architecture dictionary 
    - plot_rewards - plotting accumulated rewards  
    - encode_states - encoding of states from an integer
    - decode_positions - decodes taxi's, passenger and destinations positions from integer state
'''

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


# =============================================================================
# Set path to save images
# =============================================================================

os.chdir(os.getcwd()+'/Desktop/RL/Plots_thesis/Q-learning')

# =============================================================================
# 1. Actor-Critic 
# =============================================================================
class AC_model(nn.Module):
    def __init__(self, architecture):
        super(AC_model, self).__init__()

        num_states, hidden_units, num_actions = unpack_arch(architecture)

        # Define shared network - action head and value head
        self.hidden = nn.Linear(num_states, hidden_units)
        self.action_head = nn.Linear(hidden_units, num_actions)
        self.value_head = nn.Linear(hidden_units, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.hidden(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def select_action(self, state):
        """
            This function selects action which is sampled from the
            action head of the network (policy) - a~pi(:,s)
        :param state: state of the environment
        :return: action
        """

        # One hot encoding
        state_one_hot = np.zeros((1, self.hidden.in_features))
        state_one_hot[0, int(state)] = 1

        state = torch.from_numpy(state_one_hot).float()
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def update_weights(self, optimizer):
        """
            This function applies optimization step to the network based on a trajectory
            where the loss is the sum of minus the expected return and the squared TD
            error for the value function, V
        """

        gamma = 1   # Finite horizon
        eps = np.finfo(np.float32).eps.item()  # For stabilization

        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        rewards = []

        for r in self.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)    # Normalize - stabilize

        # Calculate losses
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value.squeeze(dim=1), torch.tensor([r])))

        # Apply optimization step
        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        optimizer.step()
        # Empty history of last trajectory
        del self.rewards[:]
        del self.saved_actions[:]


def unpack_arch(architecture):
    """
    :param architecture: dict. containing num. of units in every layer (2-layer FC network)
    :return: input_size, hidden_size, output_size of NN
    """

    input_size = architecture["state_dim"]
    hidden_size = architecture["hidden_units"]
    output_size = architecture["num_actions"]
    return input_size, hidden_size, output_size


def plot_rewards(reward_arr, avg_reward_arr, stdev_reward_arr, save=True):
    
    plt.figure(figsize=(12,9)) 
    
    fig1 = plt.figure(1)
    # rewards + average rewards
    plt.plot(reward_arr, color='blue', alpha=0.3)
    plt.plot(avg_reward_arr, color='blue')
    plt.xlabel('Number of episodes')
    plt.ylabel('Episodic reward')
    plt.title('Actor-Critic training reward per episode',size = BIGGER_SIZE)
    # plt.legend(['Acc. episodic reward', 'Avg. acc. episodic reward'])
    plt.legend([r'$\gamma$ = 0.7, $\eta$ = 0.003'], loc = 'lower right')
    plt.tight_layout()

    plt.pause(0.01)

    if save:
        fig1.savefig('AccRewardVsEpisode_AC_finite_150', dpi = 600)
        np.save('rewards_AC_finite', reward_arr)
        np.save('avg_rewards_AC_finite', avg_reward_arr)
        np.save('stdev_rewards_AC_finite', stdev_reward_arr)

    fig1.clf()


def encode_states(states, encode_method, state_dim):
    """
        Gets a list of integers and returns their encoding
        as 1 of 2 possible encoding methods:
            - one-hot encoding (array)
            - position encoding
    :param states: list of integers in [0,num_states-1]
    :param encode_method: one of 'one_hot', 'positions'
    :param state_dim: dimension of state (used for 'one_hot' encoding)
    :return: states_encoded: one hot encoding of states
    """

    batch_size = len(states)

    if encode_method is 'positions':
        '''
            position encoding encodes the important game positions as 
            a 19-dimensional vector:
                - 5 dimensions are used for one-hot encoding of the taxi's row (0-4)
                - 5 dimensions are used for one-hot encoding of the taxi's col (0-4)
                - 5 dimensions are used for one-hot encoding of the passenger's position:
                    0 is 'R', 1 is 'G', 2 is 'Y', 3 is 'B' and 4 is if the passenger in the taxi
                - 4 dimensions are used for one-hot encoding of the destination location:
                    0 is 'R', 1 is 'G', 2 is 'Y' and 3 is 'B'
                we simply concatenate those vectors into a 19-dim. vector with 4 ones in it
                corresponding to the positions encoding and the rest are zeros.      
        '''

        taxi_row, taxi_col, pass_code_1, dest_loc_1, pass_code_2, dest_loc_2 = decode_positions(states)

        # one-hot encode taxi's row
        taxi_row_onehot = np.zeros((batch_size, 5))
        taxi_row_onehot[np.arange(batch_size), taxi_row] = 1
        # one-hot encode taxi's col
        taxi_col_onehot = np.zeros((batch_size, 5))
        taxi_col_onehot[np.arange(batch_size), taxi_col] = 1
        # one-hot encode row
        pass_code_1_onehot = np.zeros((batch_size, 6))
        pass_code_1_onehot[np.arange(batch_size), pass_code_1] = 1
        # one-hot encode row
        dest_loc_1_onehot = np.zeros((batch_size, 5))
        dest_loc_1_onehot[np.arange(batch_size), dest_loc_1] = 1
        # one-hot encode row
        pass_code_2_onehot = np.zeros((batch_size, 6))
        pass_code_2_onehot[np.arange(batch_size), pass_code_1] = 1
        # one-hot encode row
        dest_loc_2_onehot = np.zeros((batch_size, 5))
        dest_loc_2_onehot[np.arange(batch_size), dest_loc_1] = 1

        states_encoded = np.concatenate([taxi_row_onehot, taxi_col_onehot,
                                         pass_code_1_onehot, dest_loc_1_onehot,
                                         pass_code_2_onehot, dest_loc_2_onehot], axis=1)

    else:   # one-hot
        states_encoded = np.zeros((batch_size, state_dim))
        states_encoded[np.arange(batch_size), states] = 1

    return states_encoded


def decode_positions(states):
    """
    Gets a state from env.render() (int) and returns
    the taxi position (row, col), the passenger position
    and the destination location
    :param states: a list of states represented as integers [0-499]
    :return: taxi_row, taxi_col, pass_code, dest_idx
    """
    dest_loc_1 = [state % 5 for state in states]
    states = [state // 5 for state in states]
    pass_code_1 = [state % 6 for state in states]
    states = [state // 6 for state in states]
    dest_loc_2 = [state % 5 for state in states]
    states = [state // 5 for state in states]
    pass_code_2 = [state % 6 for state in states]
    states = [state // 6 for state in states]
    taxi_col = [state % 5 for state in states]
    states = [state // 5 for state in states]
    taxi_row = states
    return taxi_row, taxi_col, pass_code_1, dest_loc_1, pass_code_2, dest_loc_2


# =============================================================================
# 2. Optimize actor-critic model
# =============================================================================

def objective(trial):
    """ Train the model and optimise
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    
    Args:
        trial: single execution of the objective function internally 
               instantiated upon each invocation of the function.
               
    Returns:
         Value of the mean reward obtained over the last 10 episodes
    """
    
    env = gym.make('Delivery-v0')
    
    print_every = 1000
    
    gamma = trial.suggest_discrete_uniform('gamma', 0.7, 0.8, 0.1)
    learning_rate = trial.suggest_uniform('learning_rate', 5e-4, 5e-3)
    hidden_units = trial.suggest_int('hidden units', 32,64,32)
    
    hidden_units = hidden_units
    
    gamma = gamma
    
    
    save_model = False
    save_fig = False
    
    # Define architecture parameters
    architecture = {"state_dim": env.observation_space.n,
                "hidden_units": hidden_units,
                "num_actions": env.action_space.n}
    
    # Initialize AC model
    AC_net = AC_model(architecture)
    # Define optimizer
    optimizer = optim.Adam(AC_net.parameters(), lr=learning_rate)
    
    episodes_passed = 0
    acc_rewards = []
    last_t = 0
    state = env.reset()
    
    # Initialize episodic reward list
    episodic_rewards = []
    avg_episodic_rewards = []
    stdev_episodic_rewards = []
    acc_episodic_reward = 0.0
    best_avg_episodic_reward = -np.inf
    
    
    for t in count():
    
        if len(avg_episodic_rewards) > 0:   # so that avg_episodic_rewards won't be empty
            # Stop if max episodes or playing good (above avg. reward of 5 over last 10 episodes)
            # if episodes_passed == 5000 or avg_episodic_rewards[-1] > 5:
            if episodes_passed == 50000:
                break
    
        action = AC_net.select_action(state)   # Take action
        state, reward, done, _ = env.step(action)   # Get transition
        AC_net.rewards.append(reward)               # Document reward
        acc_episodic_reward = acc_episodic_reward + reward  # Document accumulated episodic reward
        
        # Episode ends - reset environment and document statistics
        if done == True:
            episodes_passed += 1
            episodic_rewards.append(acc_episodic_reward)
            acc_episodic_reward = 0.0
        
            # Compute average reward and variance (standard deviation)
            if len(episodic_rewards) <= 10:
                avg_episodic_rewards.append(np.mean(np.array(episodic_rewards)))
                if len(episodic_rewards) >= 2:
                    stdev_episodic_rewards.append(np.std(np.array(episodic_rewards)))
        
            else:
                avg_episodic_rewards.append(np.mean(np.array(episodic_rewards[-10:])))
                stdev_episodic_rewards.append(np.std(np.array(episodic_rewards[-10:])))
        
            # Check if average acc. reward has improved
            if avg_episodic_rewards[-1] > best_avg_episodic_reward:
                best_avg_episodic_reward = avg_episodic_rewards[-1]
                if save_model:
                    torch.save(AC_net, 'trained_AC_model')
        
            # Update plot of acc. rewards every 20 episodes and print training details
            if episodes_passed % print_every == 0:
                # plot_rewards(np.array(episodic_rewards), np.array(avg_episodic_rewards),
                #               np.array(stdev_episodic_rewards), save_fig)
                print('Episode {}\tLast episode length: {:5d}\tAvg. Reward: {:.2f}\t'.format(
                    episodes_passed, t - last_t, avg_episodic_rewards[-1]))
                print('Best avg. episodic reward:', best_avg_episodic_reward)
        
            last_t = t  # Follow episodes length
            state = env.reset()
            AC_net.update_weights(optimizer)    # Perform network weights update
            
            continue
    
    last_reward = np.mean(episodic_rewards)
    return -1 * last_reward


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5, n_jobs=1)


trials_df = study.trials_dataframe()
pd.DataFrame(trials_df).to_csv("A2C_trials_d.csv")

print(df)

# =============================================================================
# 3a. Retrain the best model
# =============================================================================

env = gym.make('Delivery-v0')

print_every = 1000
hidden_units = 32
gamma = 0.7
save_model = False
save_fig = True

# Define architecture parameters
architecture = {"state_dim": env.observation_space.n,
                "hidden_units": hidden_units,
                "num_actions": env.action_space.n}

# Initialize AC model
AC_net = AC_model(architecture)
# Define optimizer
optimizer = optim.Adam(AC_net.parameters(), lr=3e-3)

episodes_passed = 0
acc_rewards = []
last_t = 0
state = env.reset()

# Initialize episodic reward list
episodic_rewards = []
avg_episodic_rewards = []
stdev_episodic_rewards = []
acc_episodic_reward = 0.0
best_avg_episodic_reward = -np.inf


for t in count():

    if len(avg_episodic_rewards) > 0:   # so that avg_episodic_rewards won't be empty
        # Stop if max episodes or playing good (above avg. reward of 5 over last 10 episodes)
        # if episodes_passed == 5000 or avg_episodic_rewards[-1] > 5:
        if episodes_passed == 150000:
            break

    action = AC_net.select_action(state)   # Take action
    state, reward, done, _ = env.step(action)   # Get transition
    AC_net.rewards.append(reward)               # Document reward
    acc_episodic_reward = acc_episodic_reward + reward  # Document accumulated episodic reward

    # Episode ends - reset environment and document statistics
    if done == True:
    # if done:
        episodes_passed += 1
        episodic_rewards.append(acc_episodic_reward)
        acc_episodic_reward = 0.0

        # Compute average reward and variance (standard deviation)
        if len(episodic_rewards) <= 10:
            avg_episodic_rewards.append(np.mean(np.array(episodic_rewards)))
            if len(episodic_rewards) >= 2:
                stdev_episodic_rewards.append(np.std(np.array(episodic_rewards)))

        else:
            avg_episodic_rewards.append(np.mean(np.array(episodic_rewards[-10:])))
            stdev_episodic_rewards.append(np.std(np.array(episodic_rewards[-10:])))

        # Check if average acc. reward has improved
        if avg_episodic_rewards[-1] > best_avg_episodic_reward:
            best_avg_episodic_reward = avg_episodic_rewards[-1]
            if save_model:
                PATH = "state_dict_model.pt"
                torch.save(AC_net.state_dict(), PATH)

        # Update plot of acc. rewards every 20 episodes and print
        # training details
        if episodes_passed % print_every == 0:
            plot_rewards(np.array(episodic_rewards), np.array(avg_episodic_rewards),
                         np.array(stdev_episodic_rewards), save_fig)
            print('Episode {}\tLast episode length: {:5d}\tAvg. Reward: {:.2f}\t'.format(
                episodes_passed, t - last_t, avg_episodic_rewards[-1]))
            print('Best avg. episodic reward:', best_avg_episodic_reward)

        last_t = t  # Follow episodes length
        state = env.reset()
        AC_net.update_weights(optimizer)    # Perform network weights update
        continue
    
# =============================================================================
# 3b. Save model
# =============================================================================
    
os.chdir(os.getcwd()+'/Desktop/RL/Plots_thesis/A2C')

PATH = "state_dict_model_150k_v2.pt"

torch.save(AC_net.state_dict(),PATH)

# =============================================================================
# 4. Load model
# =============================================================================

env = gym.make('Delivery-v0')

hidden_units = 32

# Define architecture parameters
architecture = {"state_dim": env.observation_space.n,
                "hidden_units": hidden_units,
                "num_actions": env.action_space.n}

# Initialize AC model
trained_AC_model = AC_model(architecture)

PATH = "state_dict_model_100k_v2.pt"

trained_AC_model.load_state_dict(torch.load(PATH))

print(trained_AC_model)

# =============================================================================
# 5. Evaluate model and plot results
# =============================================================================

# Reset env
state = env.reset()
# number of test episodes
num_test_episodes = 100

episodes_passed = 0
acc_episodic_reward = 0.0
episode_reward = []
frames = []

while episodes_passed < num_test_episodes:
    # Choose action greedily
    action = trained_AC_model.select_action(state)
    # Act on env
    state, reward, done, _ = env.step(action)
    # Add to accumulative reward
    acc_episodic_reward += reward
  
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


plt.figure(figsize=(12,9)) 
plt.plot(episode_reward, color = 'turquoise',label = r'$\gamma$ = 0.7, $\eta$ = 0.003')
plt.xlabel('Number of episodes')
plt.ylabel('Rewards')
plt.xlim(0, 100)
plt.axhline(0, linestyle = "--", c = "black")
plt.axhline(np.mean(episode_reward), linestyle = "--", c = "red", label = 'mean of ' + str(np.mean(episode_reward)))
plt.title('Testing reward per episode')
plt.xticks(fontsize=10)
plt.legend(loc = 'lower right')
plt.savefig("AC_testing_100.png", dpi=600)




# =============================================================================
# 6. Rendering of special cases
# =============================================================================
# Some states such as state 12, 19094 or 22125 are promising cases of 
# ride-sharing. They were identified manually in the Q-learning algorithm part.
# To determine whether the agent uses the opportunity to pick up
# both passengers at the same time, we decide to print the episode starting
# from state 12, 19094 or 22125 until the end. 

#Definition of print_frame function, the same function as in Q-learning
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
        time.sleep(1)
        

# Reset env
state = env.reset()
# Present trained behaviour over episodes
num_test_episodes = 1

episodes_passed = 0
acc_episodic_reward = 0.0
episode_reward = []
frames = []

env.s = 12

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
