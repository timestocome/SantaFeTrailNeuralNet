# https://en.wikipedia.org/wiki/Santa_Fe_Trail_problem
# https://github.com/timestocome

# adapted from https://github.com/dennybritz/reinforcement-learning gridworld example




#!/python

import numpy as np
import sys

from SantaFe import SantaFeEnv

env = SantaFeEnv()

np.set_printoptions(threshold=np.inf)   # lets us print it all to screen



"""
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: lambda discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
"""
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)



# pretty print v
grid = np.reshape(v, (32,32))
  
                
for i in range(0, 32):
    print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" %
            (grid[i][0], grid[i][1], grid[i][2], grid[i][3], grid[i][4], grid[i][5], grid[i][6], grid[i][7], grid[i][8], 
            grid[i][9], grid[i][10], grid[i][11], grid[i][12], grid[i][13], grid[i][14], grid[i][15], grid[i][16], grid[i][17], 
            grid[i][18], grid[i][19], grid[i][20], grid[i][21], grid[i][22], grid[i][23], grid[i][24], grid[i][25], grid[i][26], 
            grid[i][27], grid[i][28], grid[i][29], grid[i][30], grid[i][31]))


