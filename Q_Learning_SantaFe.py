# http://github.com/timestocome
# Trying policy iteration on the SantaFe ant maze


# solution from Denny Britz Reinforcement Learning
# https://github.com/dennybritz/reinforcement-learning
# Sutton 'Reinforcement Learning chapter 6,7,12'



import numpy as np
import sys
import pprint
import matplotlib.pyplot as plt
import gym
from SantaFeWorld import SantaFeEnv


# set up
pp = pprint.PrettyPrinter(indent=2)
env = SantaFeEnv()


# constants
number_of_actions = 4       # possible moves at each state
epsilon = 0.10
discount_factor=1.0
alpha=0.5



# see Sutton Chapter 6.4 for psuedo code
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





def sarsa(env, num_episodes):
    
    
    # The final action-value function.
    Q = np.zeros((env.nS, env.nA))
    
    # useful tracking info
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)


    # The policy we're following
    policy = make_epsilon_greedy_policy(Q)
    

    for i_episode in range(num_episodes):

        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
        
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        
        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # One step in the environment
        loop_count = 0
        max_loops = 300
        while True:

            # Take a step
            loop_count += 1
            state = env.reset()
            
            # Pick the next action
            next_action_probs = policy(state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            next_state, reward, done, _ = env.step(action)


            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = loop_count
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
    
            if done:  break
            if loop_count > max_loops: break
                
            #action = next_action
            #state = next_state        
    
    return Q, episode_lengths, episode_rewards / max_loops



Q, episode_lengths, episode_rewards = sarsa(env, 500)
print("finished")
#print("Q", Q)
#print("episode_lengths", episode_lengths)
#print("rewards", episode_rewards)


# plots 

# heatmap of Q
Q_direction = []
Q_max = []
for i in Q:
   Q_direction.append(np.argmax(i))
   Q_max.append(np.max(i)) 






Q_squared = np.asarray(Q_direction).reshape(32,32)
plt.matshow(Q_squared, interpolation='nearest', cmap='Blues')
plt.xlabel("Q SARSA Best Direction")
plt.savefig("Q_SARSA_direction.png")

plt.show()

Q_squared = np.asarray(Q_max).reshape(32,32)
plt.matshow(Q_squared, interpolation='nearest', cmap='Blues')
plt.xlabel("Q SARSA Max value")
plt.savefig("Q_SARSA_max_value.png")

plt.show()

x_values = np.arange(len(episode_rewards))
plt.plot(x_values, episode_rewards)
plt.xlabel("Rewards pver time")
plt.savefig("SARSA_Rewards.png")


plt.show() 
