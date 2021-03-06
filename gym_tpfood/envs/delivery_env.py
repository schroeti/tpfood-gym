import hashlib
import os
import random as _random
import random
from six import integer_types
import struct
import sys
import gym
from gym import Env, error, spaces, utils
from gym.utils import seeding
from gym.envs.toy_text import discrete
from contextlib import closing
from six import StringIO
import numpy as np
import scipy.stats as sps
import pandas as pd
import matplotlib.colors as mcolors


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob" : p})




MAP = [
    "+---------+",
    "|A| : : | |",
    "|:|: --- :|",
    "| : : : : |",
    "|: : --- :|",
    "| : :E: : |",
    "|: : : : :|",
    "| | : :C: |",
    "|:|: : :|:|",
    "| |B: : |D|",
    "+---------+",
]


class Delivery(DiscreteEnv):
    """
    The Multi-parcel/Multi-passenger delivery Problem

    Description:
    There are five designated locations in the grid world indicated by A, B, C, D and E. When the episode starts, the taxi/delivery man starts off at a random square and the passengers/parcels are at a random locations.
    The taxi drives to the passengers's location, picks up the passenger, drives to the passenger's destination (another one of the five specified locations), and then drops off the passenger. 
    Once both passengers/parcels have been dropped off, the episode ends.

    Observations: 
    There are 22500 discrete states since there are 25 taxi positions, 6 possible locations of the passenger 1 (including the case when the passenger is in the taxi),
    6 possible locations of the passenger 2 and 5 destination locations for passenger 1 and 5 destination locations for passenger 2. 
    
    Passenger locations:
    - 0: A
    - 1: B
    - 2: C
    - 3: D
    - 4: E
    - 5: in taxi
    
    Destinations:
    - 0: A
    - 1: B
    - 2: C
    - 3: D
    - 4: E
        
    Actions:
    There are 8 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger 1
    - 5: dropoff passenger 1
    - 6: pickup passenger 2
    - 7: dropoff passenger 2
    
    Rewards: 
    There is a reward of -1 for each action and an additional reward of +20 for delivering each passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    

    Rendering:
    - cyan (bold): passenger 1 location
    - magenta (bold): passenger 2 location
    - cyan: passenger 1 destination
    - magenta: passenger 2 destination
    - cyan (taxi): passenger 1 in taxi
    - magenta (taxi): passenger 2 in taxi 
    - green (taxi): empty taxi

    - other letters (A, B, C, D, E): locations for passengers and destinations
    

    state space is represented by:
        (taxi_row, taxi_col, passenger_location_1, passenger_destination_1, passenger_location_2, passenger_destination_2)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')
        
        num_rows = 5 #5
        num_columns = 5 #5
        

    #Create locations
        
        #self.locs = locs = [(0,0), (4,1), (3,3)] 
        self.locs = locs = [(0,0), (4,1),(3,3),(4,4),(2,2)] 
        num_states = num_rows * num_columns * (len(locs) + 1) * len(locs) * (len(locs) + 1) * len(locs) #grid size * pass_idx_1 * dest_idx_1 * pass_idx_2 * dest_idx_2
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 8
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx_1 in range(len(locs) + 1):  # +1 for being inside taxi
                    for pass_idx_2 in range(len(locs) + 1):
                        for dest_idx_1 in range(len(locs)):
                            for dest_idx_2 in range(len(locs)):
                                state = self.encode(row, col, pass_idx_1, pass_idx_2, dest_idx_1, dest_idx_2)
                                if pass_idx_1 < len(locs) and pass_idx_1 != dest_idx_1: #to say passenger is not in taxi
                                    initial_state_distrib[state] += 1
                                    
                                for action in range(num_actions):
                                    # defaults
                                    new_row, new_col, new_pass_idx_1, new_pass_idx_2 = row, col, pass_idx_1, pass_idx_2
                                    reward = -1 # default reward when there is no pickup/dropoff
                                    done = False
                                    taxi_loc = (row, col)
        
                                    if action == 0 and self.desc[2 * row + 2, 2 * col + 1] == b":" :
                                        new_row = min(row + 1, max_row)
                                        
                                    elif action == 1 and self.desc[2 * row, 2 * col + 1] == b":" : 
                                        new_row = max(row - 1, 0)
                                        
                                    if action == 2 and self.desc[2 * row + 1, 2 * col + 2] == b":" :
                                        new_col = min(col + 1, max_col)
                                        
                                    elif action == 3 and self.desc[2 * row + 1, 2 * col] == b":" :
                                        new_col = max(col - 1, 0)
                                        
                                    elif action == 4:  # pickup PASSENGER 1
                                    #passenger not in taxi, taxi in loc passenger, passenger not in destination
                                        if pass_idx_1 < len(locs) and taxi_loc == locs[pass_idx_1] and pass_idx_1 != dest_idx_1:
                                            new_pass_idx_1 = len(locs)
                                        else: # taxi is not at the passenger location or passenger already in taxi
                                            reward = -10
                                            
                                    elif action == 5:  # dropoff PASSENGER 1
                                        if taxi_loc == locs[dest_idx_1] and pass_idx_1 == len(locs):
                                            new_pass_idx_1 = dest_idx_1
                                            reward = 20
                                        # dropoff at wrong location
                                        elif (taxi_loc in locs) and pass_idx_1 == len(locs):
                                            new_pass_idx_1 = locs.index(taxi_loc)
                                        else: # non-legal dropoff because passenger not in taxi 
                                            reward = -10
                                            
                                    elif action == 6:  # pickup PASSENGER 2
                                        if pass_idx_2 < len(locs) and taxi_loc == locs[pass_idx_2] and pass_idx_2 != dest_idx_2:
                                            new_pass_idx_2 = len(locs)
                                        else: # taxi is not at the passenger location
                                            reward = -10
                                            
                                    elif action == 7:  # dropoff PASSENGER 2
                                    #if conditions fulfilled, then get 20 of reward and episode is done
                                        if taxi_loc == locs[dest_idx_2] and pass_idx_2 == len(locs):
                                            new_pass_idx_2 = dest_idx_2
                                            reward = 20
                                        # dropoff at wrong location
                                        elif (taxi_loc in locs) and pass_idx_2 == len(locs):
                                            new_pass_idx_2 = locs.index(taxi_loc)
                                        # non-legal dropoff because passenger not in taxi 
                                        else: 
                                            reward = -10
                          
                                    if new_pass_idx_1 == dest_idx_1 and new_pass_idx_2 == dest_idx_2:
                                        done = True
                                    
                                    new_state = self.encode(
                                        new_row, new_col, new_pass_idx_1, new_pass_idx_2, dest_idx_1, dest_idx_2)
                                    P[state][action].append(
                                        (1.0, new_state, reward, done))
                
        initial_state_distrib /= initial_state_distrib.sum()
        DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)
        
        
    
    def encode(self, taxi_row, taxi_col, pass_loc_1, pass_loc_2, dest_idx_1, dest_idx_2):
        i = taxi_row
        i *= 5 #5 number of rows/cols
        i += taxi_col
        i *= 6 # 5 number of passenger 1 locations
        i += pass_loc_1
        i *= 6 # 5 number of passenger 2 locations
        i += pass_loc_2
        i *= 5 #4 number of passenger 1 destinations
        i += dest_idx_1
        i *= 5 #4 number of passenger 2 destinations
        i += dest_idx_2
        return i
    
    def decode(self, i):
        out = []
        out.append(i % 5) #4 number of passenger 1 destinations
        i = i // 5  #4
        out.append(i % 5) #4 number of passenger 2 destinations
        i = i // 5  #4
        out.append(i % 6) #5 number of passenger 1 locations
        i = i // 6 #5
        out.append(i % 6) #5 number of passenger 2 locations
        i = i // 6 #5
        out.append(i % 5) #5 number of rows/cols
        i = i // 5 #5
        out.append(i)
        assert 0 <= i < 5 #5 number of rows/cols
        return reversed(out)
    
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
    
        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx_1, pass_idx_2, dest_idx_1, dest_idx_2 = self.decode(self.s)
    
        def ul(x): return "_" if x == " " else x       
        
        #passengers not in taxi
        if pass_idx_1 < len(self.locs) and pass_idx_2 < len(self.locs):
            out[2 * taxi_row + 1][2 * taxi_col + 1] = utils.colorize(
            out[2 * taxi_row + 1][2 * taxi_col + 1], 'green', highlight=True)
            
            if pass_idx_1 != dest_idx_1 and pass_idx_2 != dest_idx_2:
                pi_1, pj_1 = self.locs[pass_idx_1]
                out[2 * pi_1 + 1][2 * pj_1 + 1] = utils.colorize(out[2 * pi_1 + 1][2 * pj_1 + 1], 'cyan', bold=True)
                
                pi_2, pj_2  = self.locs[pass_idx_2]
                out[2 * pi_2 + 1][2 * pj_2 + 1] = utils.colorize(out[2 * pi_2 + 1][2 * pj_2 + 1], 'magenta', bold=True)
            
            #passenger 1 at destination and passenger 2 at origin
            elif pass_idx_1 == dest_idx_1 and pass_idx_2 != dest_idx_2:
                pi_1, pj_1 = self.locs[pass_idx_1]
                out[2 * pi_1 + 1][2 * pj_1 + 1] = utils.colorize(out[2 * pi_1 + 1][2 * pj_1 + 1], 'gray', bold=True)
                
                pi_2, pj_2  = self.locs[pass_idx_2]
                out[2 * pi_2 + 1][2 * pj_2 + 1] = utils.colorize(out[2 * pi_2 + 1][2 * pj_2 + 1], 'magenta', bold=True)
            
            #passenger 1 at origin and passenger 2 at destination
            elif pass_idx_1 != dest_idx_1 and pass_idx_2 == dest_idx_2:  
                pi_1, pj_1 = self.locs[pass_idx_1]
                out[2 * pi_1 + 1][2 * pj_1 + 1] = utils.colorize(out[2 * pi_1 + 1][2 * pj_1 + 1], 'cyan', bold=True)
                
                pi_2, pj_2  = self.locs[pass_idx_2]
                out[2 * pi_2 + 1][2 * pj_2 + 1] = utils.colorize(out[2 * pi_2 + 1][2 * pj_2 + 1], 'gray', bold=True)
           
        #passenger 1 in taxi
        elif pass_idx_1 == len(self.locs):
           out[2 * taxi_row + 1][2 * taxi_col + 1] = utils.colorize(
           out[2 * taxi_row + 1][2 * taxi_col + 1], 'cyan', highlight=True)
   
        #passenger 2 in taxi
        elif pass_idx_2 == len(self.locs):
           out[2 * taxi_row + 1][2 * taxi_col + 1] = utils.colorize(
           out[2 * taxi_row + 1][2 * taxi_col + 1], 'magenta', highlight=True)
        
        #passenger 1 AND 2 in taxi
        elif pass_idx_2 == len(self.locs):
           out[2 * taxi_row + 1][2 * taxi_col + 1] = utils.colorize(
           out[2 * taxi_row + 1][2 * taxi_col + 1], 'white', highlight=True)
        
        
        di_1, dj_1 = self.locs[dest_idx_1]
        out[2 * di_1 + 1][2 * dj_1 + 1] = utils.colorize(out[2 * di_1 + 1][2 * dj_1 + 1], 'cyan')
        
        di_2, dj_2 = self.locs[dest_idx_2]
        out[2 * di_2 + 1][2 * dj_2 + 1] = utils.colorize(out[2 * di_2 + 1][2 * dj_2 + 1], 'magenta')

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup1", "Dropoff1", "Pickup2", "Dropoff2"][self.lastaction]))
        else: outfile.write("\n")
    
        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
