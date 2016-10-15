# https://en.wikipedia.org/wiki/Santa_Fe_Trail_problem
# https://github.com/timestocome

# adapted from https://github.com/dennybritz/reinforcement-learning gridworld example




#!/python

import numpy as np
import sys

from SantaFe import SantaFeEnv

np.set_printoptions(threshold=np.inf)   # lets us print it all to screen


env = SantaFeEnv()
print("SantaFe Ant Maze has been setup")



def value_iteration(env, theta=0.0001, discount_factor=1.0):
   
    
    def one_step_lookahead(state, V):
    
        A = np.zeros(env.nA)
        A = np.random.random_integers(0, 3, [env.nA]) / 4.


        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
                print(A)
        return A
    


    #V = np.zeros(env.nS)
    V = np.random.random_integers(0, 3, [env.nS]) / 4.

    for z in range(1):                  # just a few loops for testing
    #while True:
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

"""
print("Policy Probability Distribution:")
print(policy)
print("")
"""


print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

"""
print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
"""    