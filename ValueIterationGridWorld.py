# http://github.com/timestocome
# Trying policy iteration on the SantaFe ant maze

# solution from Denny Britz Reinforcement Learning
# https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb
# Sutton 'Reinforcement Learning chapter 4'

# make sure it all works, then adjust it (different file) to work in SantaFeAnt Trail
import numpy as np
import pprint
import gym
from gridworld import GridworldEnv

# set up
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

# this time change the value function as we work through the world
def value_iteration(env, theta=0.0001, discount_factor=1.0):
   
    
    def one_step_lookahead(state, V):
        
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    V = np.zeros(env.nS)
    while True:
        # Stopping condition
        delta = 0

        # Update each state...
        for s in range(env.nS):
        
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
        
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
        
            # Update the value function
            V[s] = best_action_value        
        
        # Check if we can stop 
        if delta < theta:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    
    return policy, V



policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")