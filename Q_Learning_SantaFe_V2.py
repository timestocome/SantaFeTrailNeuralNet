# http://github.com/timestocome

# Attempt to use Deep Q Learning on SantaFe Ant problem



import numpy as np
import sys
import random
import copy
import pprint
import matplotlib.pyplot as plt


# set up
pp = pprint.PrettyPrinter(indent=2)



##############################################################################
# init
##############################################################################
# set up game board
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


maze = [
1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 
]

reward_maze = copy.copy(maze)

# constants
number_of_actions = 4       # possible moves at each state
epsilon = 0.10              # random moves
number_of_game_positions = len(maze)
learning_rate = 0.50
total_reward = 0

# create random action models for each state
models = []
for i in range(number_of_game_positions):
    r = np.random.random(4) 
    models.append(r/sum(r))



###############################################################################
# update functions
###############################################################################

# return the greedy option
def greedy_action(state):
    return np.argmax(models[state])


# return random (epsilon) action
def epsilon_action():
    return random.randint(0, 3)


# update action reward values
def update(state, action, reward):

    if reward != 0:
       
        old_actions = models[state]  # get current values
        old_actions[action] += reward * (old_actions[action] * learning_rate) # increase or leave alone depending on reward
        new_actions = old_actions / sum(old_actions)    # reset so total = 1
        models[state] = new_actions                      # save new values in model
    
        


# take a step
def step(old_state):

    # init
    new_state = None
    action = -1
    is_done = 0

    # greedy move or random exploratory move? 
    r = random.random()
    if r < epsilon:
        action = epsilon_action()
    else:
        action = greedy_action(old_state)

                         # take a step
    if action == 0:         # up
            if old_state > 31: 
                new_state = old_state - 32
    elif action == 1:       # right
            if ((old_state + 1) % 32 != 0 and old_state < 1023): 
                new_state = old_state + 1 
    elif action == 2:       # down 
            if old_state < (1023-31): 
                new_state = old_state + 32
    elif action == 3:       # left
            if (old_state % 32 != 0 and old_state > 0): 
                new_state = old_state - 1
        
    if new_state == None: new_state = old_state     # impossible move

    reward = reward_maze[new_state]                 # see if food pellet still there

    if reward != 0:
            print("New location %d" % (new_state))
            reward_maze[new_state] = 0              # update maze
            update(old_state, action, reward)       # update action values

    return new_state, reward, is_done






    

# Q learning 
def q_learning(num_episodes, discount_factor=1.0):


    # track stuff
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)   
    episode_paths = []
    


    for i_episode in range(num_episodes):
        
        # re-initalize
        path = []
        done = False
        t = 0
        reward_maze = copy.copy(maze)
        
        # pick up first pellet on start
        reward_maze[0] = 0
        total_reward = 1
        old_state = 0



        while (not done):
            
            # Choose an action and take a step

            # step 
            # init
            new_state = None
            action = -1

            # greedy move or random exploratory move? 
            r = random.random()
            if r < epsilon:
                action = epsilon_action()
            else:
                action = greedy_action(old_state)

                         # take a step
            if action == 0:         # up
                if old_state > 31: 
                    new_state = old_state - 32
            elif action == 1:       # right
                if ((old_state + 1) % 32 != 0 and old_state < 1023): 
                    new_state = old_state + 1 
            elif action == 2:       # down 
                if old_state < (1023-31): 
                    new_state = old_state + 32
            elif action == 3:       # left
                if (old_state % 32 != 0 and old_state > 0): 
                    new_state = old_state - 1
        
            if new_state == None: new_state = old_state     # impossible move

            reward = reward_maze[new_state]                 # see if food pellet still there

            if reward != 0:
                reward_maze[new_state] = 0              # update maze
                update(old_state, action, reward)       # update action values
                total_reward += reward


                       
            # Update statistics
            path.append(new_state)
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t
            
            
           
            if (t > 3096 or total_reward > 89):             # located all food or got stuck
                episode_paths.append(path)
                path = []
                break
                
            t += 1
            old_state = new_state
    
    return episode_lengths, episode_rewards, episode_paths




length, rewards, paths = q_learning(num_episodes=1000)


#print("Length stats")
#print(length)

plt.plot(rewards)
plt.title("Reward stats")
plt.show()


"""
print("Paths")
for p in paths:
    print("***********************************")
    print(p)
"""