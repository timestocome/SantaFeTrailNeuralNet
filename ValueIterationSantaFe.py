# http://github.com/timestocome
# Trying policy iteration on the SantaFe ant maze


# solution from Denny Britz Reinforcement Learning
# https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb
# Sutton 'Reinforcement Learning chapter 4'



import numpy as np
import pprint
import gym
from SantaFeWorld import SantaFeEnv





# set up
pp = pprint.PrettyPrinter(indent=2)
env = SantaFeEnv()


# see Sutton Chapter 4 for psuedo code
# this time change the value function as we work through the world
def value_iteration(env, theta=0.0001, discount_factor=1.0):
   
    

    def one_step_lookahead(state, V):
        
        # find actions with the highest value
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    V = np.zeros(env.nS)
    loop_count = 0
    while True:
        # Stopping condition
        delta = 0

        # Update each state...
        for s in range(env.nS):
        
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
        
            # Calculate delta across all states seen so far
            # adjust time value of rewards
            delta = max(delta, np.abs(best_action_value - V[s]))
        
            # Update the value function
            V[s] = best_action_value        


        # bail if not converging in a reasonable time
        # stop and save data so we can see if we're making progress despite not converging
        loop_count += 1
        if loop_count > 2000000: 
            print("bailing.....")
            break

        
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
np.savetxt("ReshapedGridPolicy2.txt", np.reshape(np.argmax(policy, axis=1), env.shape), delimiter=',')


print("Value Function:")
print(v)
print("")


print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
np.savetxt("ReshapedGridPolicy2.txt", np.reshape(np.argmax(policy, axis=1), env.shape), delimiter=',')
