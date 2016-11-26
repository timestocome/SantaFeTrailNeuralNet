# http://github.com/timestocome
# Trying policy iteration on the SantaFe ant maze

# solution from Denny Britz Reinforcement Learning
# https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb



import numpy as np
np.set_printoptions(threshold=np.nan)
import pprint
import gym
#from gridworld import GridworldEnv
from SantaFeWorld import SantaFeEnv



import datetime


print(datetime.datetime.now())




#  misc stuff

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







# set up
pp = pprint.PrettyPrinter(indent=2)
#env = GridworldEnv()
env = SantaFeEnv()

# see Sutton 4.1 Policy Evaluation for psuedo code

# evaluate a policy given a known environment
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):

    # start with all zeros in value function
    V = np.zeros(env.nS)        # nS is shape of grid, 4,4 in this case

    # while making progress....
    while True:

        delta = 0

        # for each state 
        for s in range(env.nS):     # for each state in gridworld

            v = 0                   # start fresh

            # for each possible action in this state
            for a, action_prob in enumerate(policy[s]): 

                # reward is -1 for all but final move, 0 for done
                # P is the model of the environment states and actions
                for prob, next_state, reward, done in env.P[s][a]:

                    # calculate expected value for each possible action and add to current value 
                    v += action_prob * prob * (reward + discount_factor * V[next_state])

                    # change in value function
                    delta = max(delta, np.abs(v - V[s]))
                    V[s] = v   # update value function
                    #print("V", np.sum(V))
                
                # stop once value function quits improving
                if delta < theta: break

        return np.array(V)


"""
# starting policy
# sum of actions in each state is 1.
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)


print("SantaFe value function")
#print(v.reshape(env.shape))
print_maze(v)
print(" ")

"""



def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
  
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    loop_count = 0
    
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):

            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = np.zeros(env.nA)

            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False

            policy[s] = np.eye(env.nA)[best_a]
        
        # bail if not making progress
        loop_count += 1
        if loop_count > 2000000: 
            policy_stable = True
            print("bailing.....")


        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V


policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
np.savetxt("PolicyProbabilityDistribution.txt", policy, delimiter=',')
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
np.savetxt("ReshapedGridPolicy", np.reshape(np.argmax(policy, axis=1), env.shape), delimiter=',')
print("")

print("Value Function:")
print(v)
np.savetxt("ValueFunction.txt", v, delimiter=',')
print("")


print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
np.savetxt("ReshapedGridValue.txt", v.reshape(env.shape), delimiter=",")
print("")



