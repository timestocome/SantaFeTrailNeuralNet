#/python

# https://github.com/timestocome

# Santa Fe Ant Trail with Reinforcement Learning
# http://www0.cs.ucl.ac.uk/staff/ucacbbl/bloat_csrp-97-29/node2.html
# https://en.wikipedia.org/wiki/Santa_Fe_Trail_problem
# https://github.com/timestocome

# description
# 32 x 32 grid with 90 food pellets
# minimum path requires 50 blank squares to be crossed
# minimum travel time is 90 + 50 = 140 steps
# we need 1 d array for input to network


import numpy as np


######  constants ############################################################
height = 32
width = 32

n_input = height * width      # input dimensionality i.e. game board size

up = 0                        # which direction are we facing
right = 1
down = 2
left = 3



###############################################################################################################
# set up SantaFe Ant Maze 
# and add some helper functions to make network code cleaner
np.set_printoptions(threshold=np.inf)   # lets us print it all to screen

# pretty print maze to screen
def print_maze(grid):

    grid = np.reshape(grid, (32,32))

    for i in range(0, 32):
        for j in range(0, 32):
            if grid[i][j] < 0: grid[i][j] = 0    
                
    for i in range(0, 32):
        print("%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f" %
            (grid[i][0], grid[i][1], grid[i][2], grid[i][3], grid[i][4], grid[i][5], grid[i][6], grid[i][7], grid[i][8], 
            grid[i][9], grid[i][10], grid[i][11], grid[i][12], grid[i][13], grid[i][14], grid[i][15], grid[i][16], grid[i][17], 
            grid[i][18], grid[i][19], grid[i][20], grid[i][21], grid[i][22], grid[i][23], grid[i][24], grid[i][25], grid[i][26], 
            grid[i][27], grid[i][28], grid[i][29], grid[i][30], grid[i][31]))



# pretty print maze to screen
def print_path_maze(grid):

    grid = np.reshape(grid, (32,32))

    for i in range(0, 32):
        for j in range(0, 32):
            if grid[i][j] > 0: grid[i][j] = 0    
                


    for i in range(0, 32):
        print("%4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f" %
            (grid[i][0], grid[i][1], grid[i][2], grid[i][3], grid[i][4], grid[i][5], grid[i][6], grid[i][7], grid[i][8], 
            grid[i][9], grid[i][10], grid[i][11], grid[i][12], grid[i][13], grid[i][14], grid[i][15], grid[i][16], grid[i][17], 
            grid[i][18], grid[i][19], grid[i][20], grid[i][21], grid[i][22], grid[i][23], grid[i][24], grid[i][25], grid[i][26], 
            grid[i][27], grid[i][28], grid[i][29], grid[i][30], grid[i][31]))




# init maze with food pellets laid out
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




# SantaFe rules say look at next block in the direction we are currently facing
# peak to see if food is there?
def check_for_food(x, y, d, grid):

  food = 0        # don't know if food, take a peak

  if d == up:
    if x > 0:  
      index = get_index(x-1, y)
      food = grid[index]
    return food

  if d == right:
    if y < (width-1):  
      index = get_index(x, y+1)
      food = grid[index]
    return food

  if d == down:
    if x < (height-1):  
      index = get_index(x+1, y)
      food = grid[index]
    return food

  if d == left:
    if y > 0:  
      index = get_index(x, y+1)
      food = grid[index]
    return food

  return food



# convert 2d r, c to index of 1d
def get_index(row, column):

    if row > (height - 1): row = height - 1
    if column > (width - 1): column = width - 1
    if row < 0: row = 0
    if column < 0: column = 0

    return row * width + column


# convert 1d index back to 2d index
def get_rc(index):
    c = index % height
    r = int(np.floor(index / height))
    return (r, c)



    


def move_ant(r, c, d):
    if d == 0:                              # move up
        if r > 0: 
            i = get_index(r-1, c)
            return (i, r-1, c)
        else: 
            i = get_index(r, c)
            return (i, r, c)
    elif d == 1:                            # move right
        if c < (width-1):           # check if move still inside array edges
            i = get_index(r, c+1)
            return (i, r, c+1)
        else:                   # else stay put
            i = get_index(r, c)
            return (i, r, c)
    elif d == 2:                            # move down
        if r < (height-1): 
            i = get_index(r+1, c)
            return (i, r+1, c)
        else: 
            i = get_index(r, c)
            return (i, r, c)
    elif d == 3:                            # move left
        if c > 0: 
            i = get_index(r, c-1)
            return (i, r, c-1)
        else: 
            i = get_index(r, c)
            return (i, r, c)



def get_direction(d):
    if d == 0: return "up"
    if d == 1: return "right"
    if d == 2: return "down"
    if d == 3: return "left"


m = get_maze()
print_maze(m)