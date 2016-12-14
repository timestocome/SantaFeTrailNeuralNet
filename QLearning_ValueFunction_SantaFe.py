# http://github.com/timestocome
# Trying policy iteration on the SantaFe ant maze


# solution from Denny Britz Reinforcement Learning
# https://github.com/dennybritz/reinforcement-learning/tree/master/FA
# Sutton chapter 9, 10 On Policy prediction and control 
# section 10.1, 10.2 Episodic semi-gradient control


# WIP
# currently it learns that going left gives it a reward for first 4 moves
# so it decides going left is good and only goes to the left
# changed epsilon from .1 to .5, starting to wander more helps some 
# not real happy with the way this particular algorithm is doing, 
# but there are still 4 more to work through so perhaps those will perform better?

import numpy as np
import sys
import random
import pprint
import matplotlib.pyplot as plt
import gym
from SantaFeWorld2 import SantaFeEnv


import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


# set up
pp = pprint.PrettyPrinter(indent=2)
env = SantaFeEnv()


# constants
number_of_actions = 4       # possible moves at each state
epsilon = 0.50              # random moves
discount_factor = 1.0       # 0 is short view, 1.0 is deterministic, same moves give same rewards
alpha = 0.5
epsilon_decay = 1.0


# grab some observation samples
#observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

# random observations returns state, reward
############################################################################
observation_examples = np.array([env.sample() for x in range(10000)]).astype(float)

# need 0..1 for rest of calculations but state location is 0..1023
# so change board positions from 1-1023 to -1-1


# normalize to a mean of 0 ( -1..1 )
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)


# convert from a state to a features representation using rbf kernels with different variances
featurizer = sklearn.pipeline.FeatureUnion([
    ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
    ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
    ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
    ('rbf4', RBFSampler(gamma=0.5, n_components=100))
])

# scale features
featurizer.fit(scaler.transform(observation_examples))




# value function approximator
class Estimator():


    def __init__(self):

        self.models = []

        # create a separate model for each action
        for _ in range(env.number_of_actions):

            model = SGDRegressor(learning_rate='constant')

            # initialize model for first move
            model.partial_fit([self.featurize_state(env.reset())], [0])

            self.models.append(model)


    # returns featurized representation for a state
    def featurize_state(self, state):

        state = np.array(state).reshape(1, -1)
        scaled = scaler.transform(state)
        featurized = featurizer.transform(scaled)

        return featurized[0]


    # makes value predictions
    def predict(self, s, a=None):

        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]


    # update estimates
    def update(self, s, a, y):
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])



# creates epsilon greedy policy from Q function and epsilon
def make_epsilon_greedy_policy(esimator, epsilon, nA):

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = esimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn
    

# Q learning using function approximation
def q_learning(env, estimator, num_episodes, discount_factor=1.0):

    print("q_learning")

    # track stuff
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)   
    episode_paths = []
    
    for i_episode in range(num_episodes):
        
        path = []

        # The policy we're following
        policy = make_epsilon_greedy_policy( estimator, epsilon * epsilon_decay**i_episode, env.number_of_actions)

        # Reset the environment and pick the first action
        state = env.reset()
       
        
        # One step in the environment
        #for t in itertools.count():
        done = False
        t = 0
        while (not done):
            
            # Choose an action to take
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            #print("action_probs", action_probs)
            #print("state %d, action %d" % (state[0], action))
           
            # Take a step
            next_state, reward, done = env.step(action)
            path.append(next_state)

            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t
            
            # TD Update
            q_values_next = estimator.predict((next_state, reward))
            
            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)
            
            # Update the function approximator using our target
            estimator.update(state, action, td_target)
            
            #print("\rState %d, Step %d, Episode %d Rewards %d " % (next_state, t, i_episode + 1, episode_rewards[i_episode]))
                
           
            if done | t > 2048:             # located all food or got stuck
                episode_paths.append(path)
                path = []
                break
                
            state = (next_state, reward)
            t += 1
    
    return episode_lengths, episode_rewards, episode_paths



estimator = Estimator()

length, rewards, paths = q_learning(env, estimator, 1000)


print("Length stats")
print(length)

print("Reward stats")
print(rewards)

#for p in paths:
#    print("***   ", p)
