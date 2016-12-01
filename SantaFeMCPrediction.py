# http://github.com/timestocome


# Sutton 'Reinforcement Learning' chaper 5
# applies MC Prediction to SantaFe Ant Trail
# https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Prediction%20Solution.ipynb



import numpy as np
import pprint
import gym
from SantaFeWorldMC import SantaFeEnv

from collections import defaultdict
import matplotlib.pyplot as plt
import sys



# set up
pp = pprint.PrettyPrinter(indent=2)
env = SantaFeEnv()






# maps observations to actions, returns dictionary mapping state to value
# section 5.1 Reinforcement Learning
def mc_prediction(env, num_episodes, discount_factor=1.0):


    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)


    for i_episode in range(1, num_episodes + 1):


        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()


        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        states_visited = []
        state = env.reset()

        visits = np.zeros(1024)
        rewards = np.zeros(1024)

        for t in range(100):
            actions = env.P[state]      # get all actions for this state
            probs = [actions.get(0)[0][0], actions.get(1)[0][0], actions.get(2)[0][0], actions.get(3)[0][0]] 

            action = np.random.choice(np.arange(len(probs)), p=probs)   # pick an action
            next_state, reward, done, _ = env.step(action)
            
            visits[state] += 1
            rewards[state] += reward

            if done: break
            state = next_state

        V[state] = rewards[state] / visits[state]

    
    return V    






V_10k = mc_prediction(env, num_episodes=1000)

# convert for printing
v_array = np.zeros(1024)
for k, v in enumerate(V_10k):
    v_array[k] = v

v_array = v_array.reshape(32,32)
print(v_array)

plt.matshow(v_array, interpolation='nearest', cmap='Blues')
plt.xlabel("MC_Prediction")
plt.savefig("MC_Prediction.png")

plt.show() 



