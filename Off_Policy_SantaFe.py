# http://github.com/timestocome



# calculates value function from policy using greedy policy on Q function
# Chapter 5 Sutton's 'Reinforcement Learning'
# Off policy first visit learning page 117
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



def create_random_policy():
    
    A = np.ones(number_of_actions, dtype=float) / number_of_actions
    def policy_fn(observation):
        return A
    return policy_fn





# Q maps state to action values
# returns policy with best action value 
def create_greedy_policy(Q):
   
    # A = [0.025, 0.025, 0.025, 0.025]
    # best_action from actions for this state
    # A is best action with slight chance of other actions
    def policy_fn(state):

        A = np.ones(number_of_actions, dtype=float) * epsilon / number_of_actions
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)

        return A
    
    return policy_fn





# find optimal policy using weighted sampling (greedy/random)
# return maping from state to action values
def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
   
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    rewards = np.zeros(1024)
    visits = np.zeros(1024)
    
    # The final action-value function.
    # A 2d array that maps state -> (action -> action-value).
    #Q = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = np.zeros((1024, number_of_actions))
    C = np.zeros((1024, number_of_actions))

    # The policy we're learning
    target_policy = create_greedy_policy(Q)


    
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


        sum_of_discounted_returns = 0.0
        importance_sampling_ratio = 1.0

        for t in range(len(episode)):
            
            state, action, reward = episode[t]

            sum_of_discounted_returns = discount_factor * sum_of_discounted_returns + reward
            
            # Update weighted importance sampling formula denominator
            C[state][action] += importance_sampling_ratio
            
            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (importance_sampling_ratio / C[state][action]) * (sum_of_discounted_returns - Q[state][action])
            
            # If the action taken by the behavior policy is not the action 
            # taken by the target policy the probability will be 0 and we can break
            if action !=  np.argmax(target_policy(state)):
                break
            
            importance_sampling_ratio = importance_sampling_ratio * 1./behavior_policy(state)[action]


    
    return Q, target_policy



random_policy = create_random_policy()




Q, policy = mc_control_importance_sampling(env, num_episodes=5000, behavior_policy=random_policy)




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
plt.xlabel("Off Policy")
plt.savefig("Off_Policy.png")

plt.show() 



