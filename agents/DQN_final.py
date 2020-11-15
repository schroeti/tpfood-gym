#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:45:06 2020

@author: schroeti
"""

# =============================================================================
# Import libraries
# =============================================================================


import gym
import gym_tpfood
import optuna
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.offline as py

from optuna.pruners import ThresholdPruner
from optuna import TrialPruned

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import DQN
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.bench.monitor import load_results
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy


from scipy.signal import savgol_filter

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
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# =============================================================================
# Set path to save images
# =============================================================================

os.chdir(os.getcwd()+'/Desktop/RL/Plots_thesis/DQN')


# =============================================================================
# Definition of Callback Stop Training on Reward Threshold CUSTOM
# =============================================================================

class StopPrintTrainingOnRewardThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).
    
    It must be used with the `EvalCallback`.
    
    Args:
        param reward_threshold: (float)  Minimum expected reward per episode
            to stop training.
        param check_frq: (int) Frequency of log
        param log_dir: (str) Path to the folder where the model will be saved.
          It must contains the file created by the ``Monitor`` wrapper.
        param verbose: (int),  by default = 1
        
    Returns:
        A saved model which may have reached the reward threshold.
    """
        #Define input variables
    def __init__(self, reward_threshold: float, check_freq: int, log_dir: str, verbose: int = 1):
        super(StopPrintTrainingOnRewardThreshold, self).__init__(verbose=verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        #assert self.parent is not None, ("`StopTrainingOnRewardThreshold` callback must be used "
                                         #"with an `EvalCallback`")
        
        
        # Convert np.bool to bool
        continue_training = bool(self.best_mean_reward < self.reward_threshold)
        
        #If best_mean_reward < reward_threshold, then continue training
        if self.verbose > 0 and continue_training:
             if self.num_timesteps % self.check_freq == 0:

              # Retrieve training reward
              x, y = ts2xy(load_results(self.log_dir), 'timesteps')
              if len(x) > 0:
                  # Mean training reward over the last 100 episodes
                  mean_reward = np.mean(y[-100:])
                  
                  if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
                # Print mean_reward and best_mean_reward
                  print('---------------------------------------------------------------')
                  print("Num timesteps: {}".format(self.num_timesteps))
                  print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  print('---------------------------------------------------------------')
            
        #If best_mean_reward > reward_threshold, then stop training
        if self.verbose > 0 and not continue_training:
            print("Stopping training because the mean reward {:.2f} "
                  " is above the threshold {}".format(self.best_mean_reward, self.reward_threshold))
            print("Saving new best model to {}".format(self.save_path))
            self.model.save(self.save_path)
        return continue_training

# =============================================================================
# Definition of callback CheckTraining
# =============================================================================

class CheckTraining(BaseCallback):
    """
    Check the training process by printing the log every nth episodes
    
    It must be used with the `EvalCallback`.
    
    Args:
        param check_frq: (int) Frequency of log
        param log_dir: (str) Path to the folder where the model will be saved.
          It must contains the file created by the ``Monitor`` wrapper.
        param verbose: (int),  by default = 1
        
    Returns:
        A log every nth episode
    """
    
        #Define input variables
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(CheckTraining, self).__init__(verbose=verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        
        
        #If best_mean_reward < reward_threshold, then continue training
        if self.verbose > 0:
             if self.num_timesteps % self.check_freq == 0:

              # Retrieve training reward
              x, y = ts2xy(load_results(self.log_dir), 'timesteps')
              if len(x) > 0:
                  # Mean training reward over the last 100 episodes
                  mean_reward = np.mean(y[-100:])
                  
                  if mean_reward == self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                  
                # Print mean_reward and best_mean_reward
                  print('---------------------------------------------------------------')
                  print("Num timesteps: {}".format(self.num_timesteps))
                  print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  print('---------------------------------------------------------------')


# =============================================================================
# 1. Optimization with Optuna
# =============================================================================

n_cpu = 4

def optimize_dqn(trial):
    """ Learning hyperparameters we want to optimise
    
    Args:
        trial: single execution of the objective function internally 
               instantiated upon each invocation of the function.
               
    Returns:
        gamma (float): a sampled discount rate from the discrete uniform distribution
        learning_rate (float): a sampled learning rate from the specified range
        exploration_fraction (float): a sample exploration fraction from the specified range
    
    """
    
    return {
        'gamma': trial.suggest_discrete_uniform('gamma', 0.4, 0.8, 0.2),
        'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-3, log = True), 
        'exploration_fraction': trial.suggest_float('exploration_fraction', 0.5,0.7)
    }


def optimize_agent(trial):
    """ Train and optimise the model and optimise. As Optuna maximises the
         negative log likelihood, it needs to negate the reward here. 
         
     Args: 
          trial: single execution of the objective function internally 
                instantiated upon each invocation of the function.
                
     Returns:
         Value of the mean reward obtained over the last 10 episodes
    """
    
    log_dir = "tmp/HT"
    os.makedirs(log_dir, exist_ok=True)
    
    
    
    model_params = optimize_dqn(trial)
    
    
    env = gym.make('Delivery-v0')
    env = Monitor(env, log_dir)
    model = DQN('LnMlpPolicy', env, verbose=1, double_q = True, seed = 42, buffer_size = 50000,  
                prioritized_replay = True, **model_params)
    model.learn(2000000)
        
    episodes = 10
    
    # For plotting metrics
    all_epochs = []
    rewards = []
    
    for i in range(1, episodes+1):
        obs = env.reset()
        episode_rewards = []

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            epochs += 1
        
            if done == True:
                  break 
            if epochs == 1000:
                break  
        rewards.append(np.sum(episode_rewards))

    last_reward = np.mean(rewards)
    
    #Metric to choose best reward
    return -1 * last_reward


study = optuna.create_study()
study.optimize(optimize_agent, n_trials=6, n_jobs=2)


study.best_params

#Parameters importance
importances = optuna.importance.get_param_importances(study)
fig = optuna.visualization.plot_param_importances
fig.show()


trials_df = study.trials_dataframe()
pd.DataFrame(trials_df).to_csv("DQN_trials_final_rounded.csv")

print(df)


#Extract gamma, learning_rate, and exploration fraction
best_gamma = study.best_params.get('gamma')
best_learning_rate = study.best_params.get('learning_rate')
best_exploration_fraction = study.best_params.get('exploration_fraction')


#Format dataframe
trials_df = trials_df.round({'params_exploration_fraction':3, 'params_gamma':2,'params_learning_rate':4})




# =============================================================================
# 2a. Re-Training best model DQN
# =============================================================================
from stable_baselines import DQN
env = gym.make('Delivery-v0')
env.seed(42)
log_dir = "tmp/DQN1"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(env, log_dir)

best_gamma = 0.8
best_learning_rate = 0.0019
best_exploration_fraction = 0.671


#Model 1 
time_steps = 3000000
best_model = DQN('LnMlpPolicy', env,gamma=best_gamma, exploration_fraction = best_exploration_fraction, learning_rate=best_learning_rate, buffer_size=50000, verbose=1,double_q = True, prioritized_replay = True)
callback_CT = CheckTraining(check_freq = 10000, log_dir = log_dir)


best_model.learn(total_timesteps=time_steps, callback = callback_CT)

best_model.save("dqn_best")
#best_model = DQN.load("dqn_best", env = gym.make("Delivery-v0"))


# =============================================================================
# 2b. Re-Training second model DQN (third-best model)
# =============================================================================
from stable_baselines import DQN
env = gym.make('Delivery-v0')
env.seed(42)
log_dir = "tmp/DQN2"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(env, log_dir)


#Model 2
time_steps = 3000000
model2 = DQN('LnMlpPolicy', env,gamma=0.6, exploration_fraction = 0.669, learning_rate=3e-4, buffer_size=50000, verbose=1,double_q = True, prioritized_replay = True)
callback_CT = CheckTraining(check_freq = 10000, log_dir = log_dir)


model2.learn(total_timesteps=time_steps, callback = callback_CT)


save_path_model_2 = "tmp/Model_2.zip"
model2.save(save_path_model_2)
# model2 = DQN.load(save_path_model_2)

# =============================================================================
# 2c. Re-Training worst model DQN
# =============================================================================
from stable_baselines import DQN
env = gym.make('Delivery-v0')
env.seed(42)
log_dir = "tmp/DQN3"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(env, log_dir)


#Model 3 
time_steps = 3000000
model3 = DQN('LnMlpPolicy', env,gamma=0.4, exploration_fraction = 0.502, learning_rate=1.7e-3, buffer_size=50000, verbose=1,double_q = True, prioritized_replay = True)
callback_CT = CheckTraining(check_freq = 10000, log_dir = log_dir)


model3.learn(total_timesteps=time_steps, callback = callback_CT)


save_path_model_3 = "tmp/Model_3.zip"
model3.save(save_path_model_3)
# model3 = DQN.load(save_path_model_3)


# =============================================================================
# 3. Testing best model
# =============================================================================

best_model = DQN.load("dqn_best", env = gym.make("Delivery-v0"))

env = gym.make("Delivery-v0")


def evaluate_policy_cus(model, env, n_eval_episodes = 500):
    """ Evaluate the policy of a given model
    
    Args:
        model: trained and saved DQN model
        env: environment
        n_eval_episodes (int): number of episodes to evaluate the model on
               
    Returns:
        mean_reward (float): mean reward over the episodes
        std_reward (float): standard deviation over the episodes
        episode_rewards (float): array of the episodic rewards
        episode_lengths (float): array of the number of steps per episode
    
    """
    
    env = env
    model = model
    episode_rewards, episode_lengths = [], []
    for i in range(n_eval_episodes):     
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state)
            new_obs, reward, done, _info = env.step(action)
            obs = new_obs
            episode_reward += reward
            episode_length += 1
            
            if episode_length > 30:
                break
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward, episode_rewards, episode_lengths

episode_rewards_best = evaluate_policy_cus(best_model, env = gym.make("Delivery-v0"))
episode_rewards_best = episode_rewards_best[2]


# =============================================================================
# 4.Plots
# =============================================================================

def plot_results_cus(log_folder, n, title, ep_or_t = 'timesteps'):
    """
    Plot the results given the number of models for the full number of episodes
    
    Args: 
        log_folder (str): list of the saved location of the results to plot
        n (int): number of results to plot
        title (str): the title of the task to plot
        ep_or_t(str): whether x-axis should represent timesteps or episodes
        
    Returns:
        A saved plot object of the rewards vs the number of episodes for the number of 
        specified models
    """
        
    if n == 1:
        x1, y1 = ts2xy(load_results(log_folder[0]), ep_or_t)
        y1 = moving_average(y1, window=50)
        x1 = x1[len(x1) - len(y1):]
        fig = plt.figure(title)
        plt.figure(figsize=(8,6)) 
        plt.plot(x1, y1, color = 'purple')
        plt.xlabel('Number of ' + str(ep_or_t))
        plt.ylabel('Rewards')
        plt.axhline(0, linestyle = "--", c = "black")
        plt.title(title,pad = 30 , size = 18)
        plt.legend([r'$\gamma$ = 0.8, $\eta$ = 0.0005, $\rho$ = 0.5'
            ], loc = 'lower right')
        plt.savefig(str(title)+".png", dpi=600)


    if n == 2:
        x1, y1 = ts2xy(load_results(log_folder[0]), ep_or_t)
        x2, y2 = ts2xy(load_results(log_folder[1]), ep_or_t)
        
        y1 = moving_average(y1, window=50)
        y2 = moving_average(y2, window=50)
       
        x1 = x1[len(x1) - len(y1):]
        x2 = x2[len(x2) - len(y2):]
       
        fig = plt.figure(title)
        plt.figure(figsize=(8,6)) 
        plt.plot(x1, y1, color = 'purple')
        plt.plot(x2, y2, color = 'orange')
        plt.xlabel('Number of '+ str(ep_or_t))
        plt.ylabel('Rewards')
        plt.axhline(0, linestyle = "--", c = "black")
        plt.title(title)
        plt.xticks(fontsize=10)
        plt.legend([r'$\gamma$ = 0.95, $\eta$ = 0.005,$\rho$ = 0.5 ',
            r' $\gamma$ = 0.9, $\eta$ = 0.005,$\rho$ = 0.5'
            ], loc = 'lower right')
        plt.savefig(str(title)+".png", dpi=600) 

    if n == 3:
        x1, y1 = ts2xy(load_results(log_folder[0]), ep_or_t)
        x2, y2 = ts2xy(load_results(log_folder[1]), ep_or_t)
        x3, y3 = ts2xy(load_results(log_folder[2]), ep_or_t)
        
        y1 = moving_average(y1, window=50)
        y2 = moving_average(y2, window=50)
        y3 = moving_average(y3, window=50)
       
        x1 = x1[len(x1) - len(y1):]
        x2 = x2[len(x2) - len(y2):]
        x3 = x3[len(x3) - len(y3):]
       
        fig = plt.figure(title)
        plt.figure(figsize=(8,6)) 
        plt.plot(x1, y1, color = 'purple')
        plt.plot(x2, y2, color = 'orange')
        plt.plot(x3, y3, color = 'cyan')
        plt.axhline(0, linestyle = "--", c = "black")
        plt.xlabel('Number of ' + str(ep_or_t))
        plt.ylabel('Rewards')
        plt.title(title)
        plt.xticks(fontsize=10)
        plt.legend([r' $\gamma$ = 0.8, $\eta$ = 0.002, $\rho$ = 0.67',
            r' $\gamma$ = 0.6, $\eta$ = 0.0003, $\rho$ = 0.67',
            r' $\gamma$ = 0.4, $\eta$ = 0.017, $\rho$ = 0.5'
            ], loc = 'lower right')
        plt.savefig(str(title)+".png", dpi=600) 
       
    
    if n == 4:
        x1, y1 = ts2xy(load_results(log_folder[0]), ep_or_t)
        x2, y2 = ts2xy(load_results(log_folder[1]), ep_or_t)
        x3, y3 = ts2xy(load_results(log_folder[2]), ep_or_t)
        x4, y4 = ts2xy(load_results(log_folder[3]), ep_or_t)
        
        y1 = moving_average(y1, window=50)
        y2 = moving_average(y2, window=50)
        y3 = moving_average(y3, window=50)
        y4 = moving_average(y4, window=50)
       
        x1 = x1[len(x1) - len(y1):]
        x2 = x2[len(x2) - len(y2):]
        x3 = x3[len(x3) - len(y3):]
        x4 = x4[len(x4) - len(y4):]
       
        fig = plt.figure(title)
        plt.figure(figsize=(8,6)) 
        plt.plot(x1, y1, color = 'purple')
        plt.plot(x2, y2, color = 'orange')
        plt.plot(x3, y3, color = 'cyan')
        plt.plot(x4, y4, color = 'teal')
        plt.xlabel('Number of ' + str(ep_or_t))
        plt.ylabel('Rewards')
        plt.axhline(0, linestyle = "--", c = "black")
        plt.title(title)
        plt.xticks(fontsize=10)
        plt.legend([r' $\gamma$ = 0.95, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5'
            ], loc = 'lower right')
        plt.savefig(str(title)+".png", dpi=600) 
    
    
    if n == 5:
        x1, y1 = ts2xy(load_results(log_folder[0]), ep_or_t)
        x2, y2 = ts2xy(load_results(log_folder[1]), ep_or_t)
        x3, y3 = ts2xy(load_results(log_folder[2]), ep_or_t)
        x4, y4 = ts2xy(load_results(log_folder[3]), ep_or_t)
        x5, y5 = ts2xy(load_results(log_folder[4]), ep_or_t)
        
        y1 = moving_average(y1, window=50)
        y2 = moving_average(y2, window=50)
        y3 = moving_average(y3, window=50)
        y4 = moving_average(y4, window=50)
        y5 = moving_average(y5, window=50)
       
        x1 = x1[len(x1) - len(y1):]
        x2 = x2[len(x2) - len(y2):]
        x3 = x3[len(x3) - len(y3):]
        x4 = x4[len(x4) - len(y4):]
        x5 = x5[len(x5) - len(y5):]
       
        fig = plt.figure(title)
        plt.figure(figsize=(8,6)) 
        plt.plot(x1, y1, color = 'purple')
        plt.plot(x2, y2, color = 'orange')
        plt.plot(x3, y3, color = 'cyan')
        plt.plot(x4, y4, color = 'teal')
        plt.plot(x5, y5, color = 'lime')
        plt.xlabel('Number of ' + str(ep_or_t))
        plt.ylabel('Rewards')
        plt.axhline(0, linestyle = "--", c = "black")
        plt.title(title)
        plt.xticks(fontsize=10)
        plt.legend([r' $\gamma$ = 0.95, $\eta$ = 0.005,$\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5'
            ], loc = 'lower right')
        plt.savefig(str(title)+".png", dpi=600)  


    
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    
    Args:
    values: (numpy array)
    window: (int)
    
    Returns:
    A numpy array
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

#Need to create a list of paths
log_folder = ["tmp/DQN1", "tmp/DQN2", "tmp/DQN3"]


plot_results_cus(log_folder, 3, "DQN training reward per episode ", "episodes")



def plot_results_cus_lim(log_folder, n, title, x_min , x_max, y_min, y_max, ep_or_t = 'timesteps'):
    """
    Plot the results given the number of models for the full number of episodes choosing the axes limits
    
    
    Args:
        log_folder (str): list of the saved location of the results to plot
        n (int): number of results to plot
        title (str): the title of the task to plot
        x_min (int): minimum value of x-axis
        x_max (int): maximum value of x-axis
        y_min (int): minimum value of y-axis
        y_max (int): maximum value of y-axis
        ep_or_t:(str) whether x-axis should represent timesteps or episodes
    
    
    Returns:
        A saved plot object of the rewards vs the number of episodes for the number of 
        specified models for a specified set of axes
    """

    
        
    if n == 1:
        x1, y1 = ts2xy(load_results(log_folder[0]), ep_or_t)
        y1 = moving_average(y1, window=50)
        x1 = x1[len(x1) - len(y1):]
        fig = plt.figure(title)
        
        plt.figure(figsize=(8,6)) 
        plt.plot(x1, y1, color = 'purple')
        plt.xlabel('Number of ' + str(ep_or_t))
        plt.ylabel('Rewards')
        plt.axhline(0, linestyle = "--", c = "black")
        plt.title(title)
        plt.xticks(fontsize=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min,y_max)
        plt.legend([r'$\gamma$ = 0.8, $\eta$ = 0.0005, $\rho$ = 0.5'
            ], loc = 'lower right')
        plt.savefig(str(title)+".png", dpi=600)

    if n == 2:
        x1, y1 = ts2xy(load_results(log_folder[0]), ep_or_t)
        x2, y2 = ts2xy(load_results(log_folder[1]), ep_or_t)
        
        y1 = moving_average(y1, window=50)
        y2 = moving_average(y2, window=50)
       
        x1 = x1[len(x1) - len(y1):]
        x2 = x2[len(x2) - len(y2):]
       
        fig = plt.figure(title)
        
        plt.figure(figsize=(8,6)) 
        plt.plot(x1, y1, color = 'purple')
        plt.plot(x2, y2, color = 'orange')
        plt.xlabel('Number of '+ str(ep_or_t))
        plt.ylabel('Rewards')
        plt.axhline(0, linestyle = "--", c = "black")
        plt.title(title)
        plt.xticks(fontsize=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min,y_max)
        plt.legend([r'$\gamma$ = 0.95, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5'
            ], loc = 'lower right')
        plt.savefig(str(title)+".png", dpi=600) 

    if n == 3:
        x1, y1 = ts2xy(load_results(log_folder[0]), ep_or_t)
        x2, y2 = ts2xy(load_results(log_folder[1]), ep_or_t)
        x3, y3 = ts2xy(load_results(log_folder[2]), ep_or_t)
        
        y1 = moving_average(y1, window=50)
        y2 = moving_average(y2, window=50)
        y3 = moving_average(y3, window=50)
       
        x1 = x1[len(x1) - len(y1):]
        x2 = x2[len(x2) - len(y2):]
        x3 = x3[len(x3) - len(y3):]
       
        fig = plt.figure(title)
        
        plt.figure(figsize=(8,6)) 
        plt.plot(x1, y1, color = 'purple')
        plt.plot(x2, y2, color = 'orange')
        plt.plot(x3, y3, color = 'cyan')
        plt.axhline(0, linestyle = "--", c = "black")
        plt.xlabel('Number of ' + str(ep_or_t))
        plt.ylabel('Rewards')
        plt.title(title)
        plt.xticks(fontsize=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min,y_max)
        plt.legend([r' $\gamma$ = 0.8, $\eta$ = 0.002, $\rho$ = 0.67',
            r' $\gamma$ = 0.6, $\eta$ = 0.0003, $\rho$ = 0.67',
            r' $\gamma$ = 0.4, $\eta$ = 0.017, $\rho$ = 0.5'
            ], loc = 'lower right')
        plt.savefig(str(title)+".png", dpi=600) 
       
    
    if n == 4:
        x1, y1 = ts2xy(load_results(log_folder[0]), ep_or_t)
        x2, y2 = ts2xy(load_results(log_folder[1]), ep_or_t)
        x3, y3 = ts2xy(load_results(log_folder[2]), ep_or_t)
        x4, y4 = ts2xy(load_results(log_folder[3]), ep_or_t)
        
        y1 = moving_average(y1, window=50)
        y2 = moving_average(y2, window=50)
        y3 = moving_average(y3, window=50)
        y4 = moving_average(y4, window=50)
       
        x1 = x1[len(x1) - len(y1):]
        x2 = x2[len(x2) - len(y2):]
        x3 = x3[len(x3) - len(y3):]
        x4 = x4[len(x4) - len(y4):]
       
        fig = plt.figure(title)
        
        plt.figure(figsize=(8,6)) 
        plt.plot(x1, y1, color = 'purple')
        plt.plot(x2, y2, color = 'orange')
        plt.plot(x3, y3, color = 'cyan')
        plt.plot(x4, y4, color = 'teal')
        plt.xlabel('Number of ' + str(ep_or_t))
        plt.ylabel('Rewards')
        plt.axhline(0, linestyle = "--", c = "black")
        plt.title(title)
        plt.xticks(fontsize=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min,y_max)
        plt.legend([r' $\gamma$ = 0.95, $\eta$ = 0.005, $\rho$ = 0.5',
            r'$\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5'
            ], loc = 'lower right')
        plt.savefig(str(title)+".png", dpi=600) 
    
    
    if n == 5:
        x1, y1 = ts2xy(load_results(log_folder[0]), ep_or_t)
        x2, y2 = ts2xy(load_results(log_folder[1]), ep_or_t)
        x3, y3 = ts2xy(load_results(log_folder[2]), ep_or_t)
        x4, y4 = ts2xy(load_results(log_folder[3]), ep_or_t)
        x5, y5 = ts2xy(load_results(log_folder[4]), ep_or_t)
        
        y1 = moving_average(y1, window=50)
        y2 = moving_average(y2, window=50)
        y3 = moving_average(y3, window=50)
        y4 = moving_average(y4, window=50)
        y5 = moving_average(y5, window=50)
       
        x1 = x1[len(x1) - len(y1):]
        x2 = x2[len(x2) - len(y2):]
        x3 = x3[len(x3) - len(y3):]
        x4 = x4[len(x4) - len(y4):]
        x5 = x5[len(x5) - len(y5):]
       
        fig = plt.figure(title)
        
        plt.figure(figsize=(8,6)) 
        plt.plot(x1, y1, color = 'purple')
        plt.plot(x2, y2, color = 'orange')
        plt.plot(x3, y3, color = 'cyan')
        plt.plot(x4, y4, color = 'teal')
        plt.plot(x5, y5, color = 'lime')
        plt.xlabel('Number of ' + str(ep_or_t))
        plt.ylabel('Rewards')
        plt.axhline(0, linestyle = "--", c = "black")
        plt.title(title)
        plt.xticks(fontsize=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min,y_max)
        plt.legend([r' $\gamma$ = 0.95, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5',
            r' $\gamma$ = 0.9, $\eta$ = 0.005, $\rho$ = 0.5'
            ], loc = 'lower right')
        plt.savefig(str(title)+".png", dpi=600)  



plot_results_cus_lim(log_folder, 3, "DQN training reward per episode zoomed",20000,30000,-100,50, ep_or_t = 'episodes')

plot_results_cus_lim(log_folder, 3, "DQN training reward per episode zoomed end",60000,100000,-100,50, ep_or_t = 'episodes')

plt.savefig('DQN_training.png', dpi=600)

