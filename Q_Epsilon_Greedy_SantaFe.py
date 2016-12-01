# http://github.com/timestocome



# calculates value function from policy using greedy policy on Q function
# Chapter 5 Sutton's 'Reinforcement Learning'
# On policy first visit learning page 107
# https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Prediction%20Solution.ipynb


import numpy as np
import sys
import matplotlib as plt

import gym
from SantaFeWorldMC import SantaFeEnv

from collections import defaultdict
import matplotlib.pyplot as plt



# set up
env = SantaFeEnv()


# max number of actions available in each state
number_of_actions = 4
epsilon = 0.10



# Q maps state to action values
# returns policy with best action value 
def make_epsilon_greedy_policy(Q):
   
    # A = [0.025, 0.025, 0.025, 0.025]
    # best_action from actions for this state
    # A is best action with slight chance of other actions
    def policy_fn(state):

        A = np.ones(number_of_actions, dtype=float) * epsilon / number_of_actions
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)

        return A
    
    return policy_fn





# find optimal policy using greedy method
# return maping from state to action values
def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0):
   
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    rewards = np.zeros(1024)
    visits = np.zeros(1024)
    
    # The final action-value function.
    # A 2d array that maps state -> (action -> action-value).
    #Q = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = np.zeros((1024, 4))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q)


    
    for i_episode in range(1, num_episodes + 1):

        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()

        for t in range(100):
        
            actions = env.P[state]      # get all actions for this state
            probs = [actions.get(0)[0][0], actions.get(1)[0][0], actions.get(2)[0][0], actions.get(3)[0][0]] 

            action = np.random.choice(np.arange(len(probs)), p=probs)   # pick an action
            next_state, reward, done, _ = env.step(action)

            visits[state] += 1     # scale for reward scaling
            rewards[state] += reward

            if done: break

            state = next_state

       
        Q[state] = rewards[state] / visits[state]
    
    return Q, policy




Q, policy = mc_control_epsilon_greedy(env, num_episodes=5000)




# For plotting: Create value function from action-value function
# by picking the best action at each state
#values = defaultdict(float)
#for state, actions in Q.items():
#    action_value = np.max(actions)
#    values[state] = action_value


values = np.zeros(1024)
for i in range(1024):
    action_values = Q[i]
    best_action = max(action_values)
    values[i] = best_action



values = values.reshape(32,32)


plt.matshow(values, interpolation='nearest', cmap='Blues')
plt.xlabel("Epsilon Greedy")
plt.savefig("Epsilon_Greedy.png")

plt.show() 



