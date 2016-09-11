#/python

# https://github.com/timestocome

# Santa Fe Ant Trail with Reinforcement Learning
# http://www0.cs.ucl.ac.uk/staff/ucacbbl/bloat_csrp-97-29/node2.html
# https://en.wikipedia.org/wiki/Santa_Fe_Trail_problem
# https://github.com/timestocome



# starter code 
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# http://karpathy.github.io/2016/05/31/rl/





# description
# 32 x 32 grid with 90 food pellets
# minimum path requires 50 blank squares to be crossed
# minimum travel time is 90 + 50 = 140 steps


# to do
# put ant on board instead of tracking location separately?
# add bias, will this help vanishing weights?
# check back propagation doesn't have matrices twisted around
# done at end of maze or every time ant locates food?
# weights vanishing 
# .... try relu? instead of sig, if change need to change grad calc?
# .... try 0s instead of -1 in maze?



import numpy as np
import pickle
import SantaFeMaze 



########################################################################################################
# tweak network parameters here:

batch_size = 10               # every how many episodes to do a param update?
learning_rate = 1e-4          # keeps weights from changing too quickly
gamma = 0.99                  # discount factor for reward, 1 == infinite look back, 0 this reward is only one that matters
decay_rate = 0.99             # decay factor leaky sum of grad^2

resume = False                # resume from previous checkpoint?
render = False                # screen display game move

number_updates_between_saves = 10  # how often to save game weights etc?
max_episodes = 10             # end the training loop after how many games?, set to 500 once things are setup

points = 0.                   # what we start with

max_moves = 600               # we've wandered lost for too long
max_rewards = 40              # we've won --- this should be 90 pellets - 50 blanks = 40 for no errors through maze

######  constants ############################################################

height = 32                   # maze dimensions
width = 32

n_input = height * width      # input dimensionality i.e. game board size
n_output = 3                  # move forward? left? right?
n_hidden = 64                 # number of hidden layer neurons
n_input_weights = n_input * n_hidden
n_output_weights = n_hidden * n_output

not_zero = 0.000001



######## reload saved state or init new ##########################################

if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  # "Xavier" initialization http://deepdish.io/2015/02/24/network-initialization/
  model = {}
  model['W1'] = np.random.randn(n_hidden, n_input) / np.sqrt(n_input)                      
  model['W2'] = np.random.randn(n_output, n_hidden) / np.sqrt(n_hidden)
  

grad_buffer = { k : np.zeros_like(v) for k, v in model.items() }     # update buffers that add up gradients over a batch
rms_prop_cache = { k : np.zeros_like(v) for k, v in model.items() }  # root mean square propagation memory


# set up SantaFe Ant Maze 
maze = SantaFeMaze.get_maze()


# food pellet setup
max_pellets = sum(maze.flatten())               # count number of food pellets available
points = 1.                                     # we begin maze on top of a food pellet 
maze[0] = 0.                                    # remove this pellet from maze
food_collected = 1                              # ant starts in a food location at (0, 0)

# tracking ant setup
previous_maze = SantaFeMaze.get_maze()          # used in computing the differences between frames
current_maze = SantaFeMaze.get_maze()


index = 0                                       # starting location
x, y = SantaFeMaze.get_rc(index)
d = np.random.randint(3)                     # random starting direction
print(x, y, d)


# track progress setup
rewards_sum = 0.
episode_number = 0.
done = 0
move_number = 0

ins, hs, output_error, rewards = [],[],[],[]    # save input, hidden output, output, rewards
hidden_output = np.zeros(n_hidden)




# moving ant setup
action = np.zeros(n_output)                     # move forward, rotate right, rotate left
action_probability = np.zeros(n_output)

# setup user views
path_maze = SantaFeMaze.get_maze()              # used for visuals


######## network helper functions ###############################################################################

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x))                                    # sigmoid "squashing" function to interval [0,1]





# take 1D float array of rewards and compute discounted reward 
def discount_rewards(r):

  # init variables
  discounted_r = np.zeros_like(r)
  running_add = 0

  # work backwards for each move
  for t in reversed(range(0, r.size)):

    # game ends with 1, -1 win/lose, 0 is on going game
    if r[t] != 0: running_add = 0                     

    running_add = running_add * gamma + r[t]       # discount rewards over time
    discounted_r[t] = running_add                  # copy to time step

  return discounted_r




# forward propagation
# push changes in game board each cycle through network
def policy_forward(diff_maze):

  i = diff_maze.flatten()                           # convert maze to 1 dimension

  h = np.dot(model['W1'], i)                        # hidden layer = x * Wi
  h[h<0] = 0                                        # ReLU nonlinearity
  
  logistic_probability = np.dot(model['W2'], h)     # logistic layer = Wo * h 
  p = sigmoid(logistic_probability)                 # smooth output with sigmoid

  # first run through adds extra dimension to h, flattening it fixes this 
  return p, h.flatten()                             # return probabilities of game positions



# work error backwards 
# episodes are the arrays containing the game states after each move
def policy_backward(episodes_h, episodes_error, episodes_diff_maze):

  loops = episodes_diff_maze.size/n_input

  # error per output weight
  dW2 = np.dot(episodes_h.T, episodes_error)         # error dotted with output weights

  # error per hidden node2  
  # dh = np.outer(episodes_error, model['W2'])
  dh = np.dot(episodes_error, model['W2'])
  dh[episodes_h < 0] = 0
   

  # error per input weight
  episodes_i = np.reshape(episodes_diff_maze, (1024, loops))
  dW1 = np.dot(dh.T, episodes_i.T)
  
  return {'W1':dW1, 'W2':dW2.T}                       # return gradients for weights







############# main loop  ####################################################################
# one step at a time work through maze keeping track of food pellets 
# at end work errors back through probabilities board and try again

# episodes are number of times to attempt to solve maze
while episode_number < max_episodes:

  
  move_number += 1                            # keep ant from wandering forever

  # #########################################################################################
  # pick up food and move forward if food in front of ant in direction ant is facing
  # else check the probabilities to decide on next move
  food = SantaFeMaze.check_for_food(x, y, d, current_maze)  



  # remove food, flag spot as somewhere we've been before, punish standing still
  # update other copies of maze
  index = SantaFeMaze.get_index(x, y)
  current_maze[index] -= 1                  # remove food or record ant visit to cell
  changes_maze = np.subtract(current_maze, previous_maze) # record changes in maze
  previous_maze[index] -= 1                 # now update previous_maze 
  path_maze[index] -= 1                     # use to view ant's travels

  
  # if food ahead move and scoop it up
  if food == 1:
     index, x, y = SantaFeMaze.move_ant(x, y, d)         # move to food location
     action[0] = 1
     points = 1.                            # track food pellets recovered
     food_collected += 1

  # no food ahead so rotate or step forward
  else:
    points = -1.                            # no food use -1, 0 breaks lots of calculations 
    
    # look at game probabilities and make a move
    action_probability, hidden_output = policy_forward(changes_maze)          # hidden layer output, and output values

    move_forward = action_probability[0]
    rotate_left = action_probability[1]
    rotate_right = action_probability[[2]]

    # the lower our action probabilities the more likely we'll try a random move    
    # the game progresses we should become more conservative
    go_forward = 0 if np.random.uniform() < move_forward else 1
    if go_forward:
        index, x, y = SantaFeMaze.move_ant(x, y, d)
        action[0] = 1
    else:
        action[0] = 0

    go_left = 0 if np.random.uniform() < rotate_left else 1
    if go_left:
      d += 1
      if d > 3: d = 0
      action[1] = 1
    else:
      action[1] = 0

    go_right = 0 if np.random.uniform() < rotate_right else 1
    if go_right:
      d -= 1
      if d < 0: d = 3
      action[2] = 1
    else:
      action[2] = 0
    
    
  #print("location (%d, %d), direction %d, rewards sum %f" % (x, y, d, rewards_sum))
  #########################################################################################

  # record various intermediates (needed later for backprop)
  ins.append(changes_maze)                         # changes in game board
  hs.append(hidden_output)                         # hidden state

  

  # grad that encourages the action that was taken to be taken 
  # http://cs231n.github.io/neural-networks-2/#losses 
  output_error.append(action - action_probability)      # action taken - network recommended action

  # step the environment and get new measurements
  # observation, reward, done, info = env.step(action)  # make the move
  rewards_sum += points                                 # add results of action into total reward
  rewards.append(rewards_sum)                             # record reward for previous action)
  
  # game over ?
  if move_number >= max_moves: 
    done = -1

  if rewards_sum >= max_rewards: 
    done = 1
  

  # adjust weights when finding food or when game over?
  if food == 1: print("found food")
  if food == 1:
  # game over
  #if done != 0: 

    done = 0
    episode_number += 1
    #print("***********************    Episode number", episode_number)

    # concat all inputs, hidden states, action gradients, and rewards for this episode
    episodes_i = np.vstack(ins)                        # inputs
    episodes_h = np.vstack(hs)                         # hiddens
    episodes_error = np.vstack(output_error)           # errors
    episodes_r = np.vstack(rewards)                    # rewards
    
    ins, hs, output_error, rewards = [],[],[],[]       # reset array memory

    # compute the discounted reward backwards through time
    discounted_episodes_r = discount_rewards(episodes_r)
    
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episodes_r -= np.mean(discounted_episodes_r)
    discounted_episodes_r /= np.std(discounted_episodes_r) 

    episodes_error *= discounted_episodes_r     # modulate the gradient with advantage (PG magic happens right here.)

  
    grad = policy_backward(episodes_h, episodes_error, episodes_i)        # compute gradient
    for k in model:grad_buffer[k] += grad[k]                              # append current gradient



    # loop over weights
    for key, value in model.items():
        g = grad_buffer[key]          # gradient for this set of weights
        rms_prop_cache[key] = decay_rate * rms_prop_cache[key] + (1 - decay_rate) * g**2 # decay gradients
        model[key] += learning_rate * g / (np.sqrt(rms_prop_cache[key]) + 1e-5) # weights += lr * gradient / sqrt(decayed gradients)
        grad_buffer[key] = np.zeros_like(value)                                 # reset batch gradient buffer        


    # sanity check weights, grads
    print("Gradients (in/out)", sum(grad['W1'].flatten()/n_input_weights), sum(grad['W2'].flatten()/n_output_weights))
    print("Weights ( in/out )? ", sum(model['W1'].flatten()/n_input_weights), sum(model['W2'].flatten()/n_output_weights))
    
      
    
    # save every so many iterations
    if episode_number % number_updates_between_saves == 0: pickle.dump(model, open('save.p', 'wb'))


    # clean up visuals
    print("***************************************************************************************************")
    print("food collected %d = %.2f%%" % (food_collected, food_collected / 90. * 100.))
    print("wandering = %.2f%%" % (rewards_sum/max_moves * 100.))
    #print_path_maze(path_maze)
    print("***************************************************************************************************")

    # reset all for next attempt
    food_collected = 0
    rewards_sum = 0
    move_number = 0

    maze = SantaFeMaze.get_maze()                   # reset maze
    previous_maze = SantaFeMaze.get_maze()             # used in computing the differences between frames
    current_maze = SantaFeMaze.get_maze()
    path_maze = SantaFeMaze.get_maze()              # used for visuals

    x = 0                                           # put ant back on first square
    y = 0

  