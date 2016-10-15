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

#    Evaluate a policy given an environment and a full description of the environment's dynamics.
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



"""
# test policy evaluation
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
"""



#    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
#    until an optimal policy is found.
def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    
    # Start with a random policy
    #policy = np.ones([env.nS, env.nA]) / env.nA
    policy = np.random.random_integers(0, 3, [env.nS, env.nA]) / 4.


    
    # test with 3 loops till the bugs are worked out
    for z in range(1):

        print("****** %d ************************************" % (z))
        # Evaluate the current policy in the environment 
        V = policy_eval_fn(policy, env, discount_factor)
        policy_stable = True
        print(V.sum())
        
        # Will be set to false if we make any changes to the policy
        #if i > 0: policy_stable = True
        #else: policy_stable = False
        
        # For each state...
        for s in range(env.nS):

            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            print("position, policy", s, policy[s])

            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    print(reward)
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False

            policy[s] = np.eye(env.nA)[best_a]
        


        # If the policy is stable we've found an optimal policy. Return it
        #if policy_stable:
        #    if i != 0:
    return policy, V
      




policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

"""

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
"""