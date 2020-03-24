import hashlib
import os
import random as _random
from six import integer_types
import struct
import sys
import gym
from gym import error, spaces, utils
from gym.envs.toy_text import discrete
from gym.utils import seeding
from contextlib import closing
from six import StringIO
import numpy as np





MAP = [
    "+-------------+",
    "|R: : : :G: : |",
    "|------. . . .|",
    "| : : : : : : |",
    "|. . . . . . .|",
    "| : : : : : : |",
    "|. -----------|",
    "| : : : : : : |",
    "|. . . . . . .|",
    "|Y| | |B| : : |",
    "|. . . . . . .|",
    "| : : : : : : |",
    "|. . . . . . .|",
    "| : : : : : :Z|",
    "+-------------+",
]


class Delivery(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.

    Observations: 
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations. 
    
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    - 5: Z
    
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: Z
        
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y, B and Z): locations for passengers and destinations
    

    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.locs = locs = [(0,0), (0,4), (4,0), (4,3), (6,6)] #Z is (6,6)
        
        num_states = 1470 #500
        num_rows = 7 #5
        num_columns = 7 #5
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 5 and pass_idx != dest_idx:
                            initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = -1 # default reward when there is no pickup/dropoff
                            done = False
                            taxi_loc = (row, col)

                            if action == 0 and self.desc[2*row+2, 2 * col + 1] == b"." :
                                new_row = min(row + 1, max_row)
                            elif action == 1 and self.desc[2*row, 2 * col + 1] == b"." : 
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":" :
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":" :
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if (pass_idx < 5 and taxi_loc == locs[pass_idx]):
                                    new_pass_idx = 5
                                else: # passenger not at location
                                    reward = -10
                            elif action == 5:  # dropoff
                                if (taxi_loc == locs[dest_idx]) and pass_idx == 5:
                                    new_pass_idx = dest_idx
                                    done = True
                                    reward = 20
                                elif (taxi_loc in locs) and pass_idx == 5:
                                    new_pass_idx = locs.index(taxi_loc)
                                else: # dropoff at wrong location
                                    reward = -1
                                
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx)
                            P[state][action].append(
                                (1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 7 #5 number of rows/cols
        i += taxi_col
        i *= 6 # 5 number of passenger locations
        i += pass_loc
        i *= 5 #4 number of passenger destinations
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 5) #4 number of passenger destinations
        i = i // 5  #4
        out.append(i % 6) #5 number of passenger locations
        i = i // 6 #5
        out.append(i % 7) #5 number of rows/cols
        i = i // 7 #5
        out.append(i)
        assert 0 <= i < 7 #5 number of rows/cols
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x): return "_" if x == " " else x
        if pass_idx < 5: #number of passenger destinations
            out[2 * taxi_row + 1][2 * taxi_col + 1] = utils.colorize(
                out[2 * taxi_row + 1][2 * taxi_col + 1], 'yellow', highlight=True)
            pi, pj = self.locs[pass_idx]
            out[2 * pi + 1][2 * pj + 1] = utils.colorize(out[2 * pi + 1][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[2 * taxi_row + 1][2 * taxi_col + 1] = utils.colorize(
                ul(out[2  * taxi_row + 1][2 * taxi_col + 1]), 'green', highlight=True)

        di, dj = self.locs[dest_idx]
        out[2 * di + 1][2 * dj + 1] = utils.colorize(out[2 * di + 1][2 * dj + 1], 'magenta')
        

        
        
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
