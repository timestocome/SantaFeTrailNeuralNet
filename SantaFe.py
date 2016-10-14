# https://en.wikipedia.org/wiki/Santa_Fe_Trail_problem
# https://github.com/timestocome

# adapted from https://github.com/dennybritz/reinforcement-learning gridworld example


    # SantaFe Ant world environment 
    # 32 x 32 grid with 90, food pellets
    # minimum path requires 50, blank squares to be crossed
    # minimum travel time is 90, + 50, = 140, steps
    # we need 1 d array for input to network
    # begin top left, finish bottom right on last one

    # T 1 1 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # . . . 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # . . . 1 . . . . . . . . . . . . . . . . . . . . . 1 1 1 . . . .
    # . . . 1 . . . . . . . . . . . . . . . . . . . . 1 . . . . 1 . .
    # . . . 1 . . . . . . . . . . . . . . . . . . . . 1 . . . . 1 . .
    # . . . 1 1 1 1 . 1 1 1 1 1 . . . . . . . . 1 1 . . . . . . . . .
    # . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . 1 . .
    # . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . .
    # . . . . . . . . . . . . 1 . . . . . . . 1 . . . . . . . . . . .
    # . . . . . . . . . . . . 1 . . . . . . . 1 . . . . . . . . 1 . .
    # . . . . . . . . . . . . 1 . . . . . . . 1 . . . . . . . . . . .
    # . . . . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . .
    # . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . 1 . .
    # . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . .
    # . . . . . . . . . . . . 1 . . . . . . . 1 . . . . . 1 1 1 . . .
    # . . . . . . . . . . . . 1 . . . . . . . 1 . . 1 . . . . . . . .
    # . . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . .
    # . . . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . .
    # . . . . . . . . . . . . 1 . . . 1 . . . . . . . 1 . . . . . . .
    # . . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . 1 . . . .
    # . . . . . . . . . . . . 1 . . . . . . . . . . . . . . . . . . .
    # . . . . . . . . . . . . 1 . . . 1 . . . . . . . . . . . . . . .
    # . . . . . . . . . . . . 1 . . . . . . . . . . . . . 1 . . . . .
    # . . . . . . . . . . . . 1 . . . . . . . . . . T . . . . . . . .
    # . . . 1 1 . . 1 1 1 1 1 . . . . 1 . . . . . . . . . . . . . . .
    # . 1 . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . .
    # . 1 . . . . . . . . . . . . . . 1 . . . . . . . . . . . . . . .
    # . 1 . . . . . . 1 1 1 1 1 1 1 . . . . . . . . . . . . . . . . .
    # . 1 . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . .
    # . . . . . . . 1 . . . . . . . . . . . . . . . . . . . . . . . .
    # . . 1 1 1 1 . . . . . . . . . . . . . . . . . . . . . . . . . .
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .



import numpy as np
import sys
from gym.envs.toy_text import discrete




UP = 0,
RIGHT = 1
DOWN = 2
LEFT = 3





pellet = 1.
blank = -1.
def get_maze():

    grid = np.zeros((32, 32))

    for i in range(0, 32):
        for j in range(0, 32):
            grid[i][j] = blank

    grid[0,0] = pellet
    grid[0,1] = pellet
    grid[0,2] = pellet
    grid[0,3] = pellet

    grid[1,3] = pellet
    grid[2,3] = pellet
    grid[3,3] = pellet
    grid[4,3] = pellet
    grid[5,3] = pellet

    grid[5,4] = pellet
    grid[5,5] = pellet
    grid[5,6] = pellet

    grid[5,8] = pellet
    grid[5,9] = pellet
    grid[5,10] = pellet
    grid[5,11] = pellet
    grid[5,12] = pellet

    grid[6,12] = pellet
    grid[7,12] = pellet
    grid[8,12] = pellet
    grid[9,12] = pellet
    grid[10,12] = pellet

    grid[12,12] = pellet
    grid[13,12] = pellet
    grid[14,12] = pellet
    grid[15,12] = pellet

    grid[18,12] = pellet
    grid[19,12] = pellet
    grid[20,12] = pellet
    grid[21,12] = pellet
    grid[22,12] = pellet
    grid[23,12] = pellet

    grid[24,11] = pellet
    grid[24,10] = pellet
    grid[24,9] = pellet
    grid[24,8] = pellet
    grid[24,7] = pellet

    grid[24,4] = pellet
    grid[24,3] = pellet

    grid[25,1] = pellet
    grid[26,1] = pellet
    grid[27,1] = pellet
    grid[28,1] = pellet

    grid[30,2] = pellet
    grid[30,3] = pellet
    grid[30,4] = pellet
    grid[30,5] = pellet

    grid[29,7] = pellet
    grid[28,7] = pellet

    grid[27,8] = pellet
    grid[27,9] = pellet
    grid[27,10] = pellet
    grid[27,11] = pellet
    grid[27,12] = pellet
    grid[27,13] = pellet
    grid[27,14] = pellet

    grid[26,16] = pellet
    grid[25,16] = pellet
    grid[24,16] = pellet

    grid[21,16] = pellet

    grid[19,16] = pellet
    grid[18,16] = pellet
    grid[17,16] = pellet

    grid[16,17] = pellet

    grid[15,20] = pellet
    grid[14,20] = pellet

    grid[11,20] = pellet
    grid[10,20] = pellet
    grid[9,20] = pellet
    grid[8,20] = pellet

    grid[5,21] = pellet
    grid[5,22] = pellet

    grid[4,24] = pellet
    grid[3,24] = pellet

    grid[2,25] = pellet
    grid[2,26] = pellet
    grid[2,27] = pellet

    grid[3,29] = pellet
    grid[4,29] = pellet

    grid[6,29] = pellet

    grid[9,29] = pellet

    grid[12,29] = pellet

    grid[14,28] = pellet
    grid[14,27] = pellet
    grid[14,26] = pellet

    grid[15,23] = pellet

    grid[18,24] = pellet

    grid[19,27] = pellet

    grid[22,26] = pellet

    grid[23,23] = pellet


    return grid.flatten()



maze = get_maze()

def get_reward(i):
    
    reward = maze[i]
    if maze[i] == 1: maze[i] = -1       # remove food pellet

    return reward


    # You can take actions in each direction (UP=0,, RIGHT=1, DOWN=2, LEFT=3).
    # Actions going off the edge leave you in your current state.
    # You receive a reward of -1 at each step, that is empty, +1 at food states
    # until all food is collected
    


class SantaFeEnv(discrete.DiscreteEnv):
   

    metadata = {'render.modes': ['human', 'ansi']}
    

    def __init__(self, shape=[32,32]):

        global max_food
        max_food = 90

        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 32

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a : [] for a in range(nA)}

            #is_done = lambda s: s == 0 or s == (nS - 1)
            #reward = 0.0 if is_done(s) else -1.0

            # was there still a food pellet on this spot
            reward = get_reward(s)
            if reward == 1: max_food -= 1

            # did we collect all the food pellets
            if max_food <= 0: is_done = True
            else: is_done = False

             # We're stuck in a terminal state
            if is_done:
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]

            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1

                P[s][UP] = [(1.0, ns_up, reward, False)]
                P[s][RIGHT] = [(1.0, ns_right, reward, False)]
                P[s][DOWN] = [(1.0, ns_down, reward, False)]
                P[s][LEFT] = [(1.0, ns_left, reward, False)]

            it.iternext()


        # Initial state distribution 
        isd = get_maze()

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(SantaFeEnv, self).__init__(nS, nA, P, isd)




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