# github.com/timestocome




import numpy as np
import random
import copy

import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3




maze = [
1, 1, 1, 1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 1, 1, 0,  0,  0,  0,
0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  1, 0,  0,
0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  1, 0,  0,
0,  0,  0,  1, 1, 1, 1, 0,  1, 1, 1, 1, 1, 0,  0,  0,  0,  0,  0,  0,  0,  1, 1, 0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  1, 1, 1, 0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  1, 1, 0,  0,  1, 1, 1, 1, 1, 0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  1, 0,  0,  0,  0,  0,  0,  1, 1, 1, 1, 1, 1, 1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  1, 0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  1, 1, 1, 1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 
]


class SantaFeEnv(discrete.DiscreteEnv):

    #     Santa Fe Ant Trail with Reinforcement Learning
    # http://www0.cs.ucl.ac.uk/staff/ucacbbl/bloat_csrp-97-29/node2.html
    # https://en.wikipedia.org/wiki/Santa_Fe_Trail_problem
    # https://github.com/timestocome

    # description
    # 32 x 32 grid with 90 food pellets
    # minimum path requires 50 blank squares to be crossed
    # minimum travel time is 90 + 50 = 140 steps
    # we need 1 d array for input to network



    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):

        nA = 4
        MAX_Y = 32
        MAX_X = 32
        shape=[32, 32]
        nS = np.prod(shape)
        grid = np.arange(nS).reshape(shape)

        self.reward_maze = copy.copy(maze)
        self.total_reward = 0
        self.state = 0
       
        # allow main program to grab information
        self.shape = shape
        self.number_of_actions = nA
        self.number_of_states = nS

        self.sample = self.sample
        self.reset = self.reset


    def sample(self):
        
        s = random.randint(0, 1023)
        reward = self.reward_maze[s]     

        return s, reward






    def step(self, action):
      
        s = self.state                   # current board location
        is_done = 0
        if self.total_reward >= 90: 
            is_done = 1
        else:                       # take a step
            if action == 0:         # up
                if s > 31: s -= 32
            elif action == 1:       # right
                if (s+1) % 32 != 0 & s < 1023: s += 1 
            elif action == 2:       # down 
                if s < (1023-31): s += 32
            elif action == 4:       # left
                if s % 32 != 0 & s > 0: s -= 1

        reward = self.reward_maze[s]     
        self.total_reward += maze[s]        # track food locations found
        self.reward_maze[s] = 0             # remove food after finding it
        self.state = s                      # update game board

        return s, reward, is_done




    def reset(self):
        self.reward_maze = copy.copy(maze)
        self.total_reward = 0
        self.state = 0

        return self.state, self.reward_maze[self.state]



    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip() 
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()