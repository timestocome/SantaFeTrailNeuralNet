# https://en.wikipedia.org/wiki/Santa_Fe_Trail_problem
# https://github.com/timestocome

# adapted from https://github.com/dennybritz/reinforcement-learning gridworld example


import gym
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict



from SantaFe import SantaFeEnv
import plotting


env = SantaFeEnv()

##############################################################################
# utility functions

def plot_maze(V):
    #print(V)
    state_map = np.zeros(1024)
    for s in V:
        state_map[s] = V[s]

    print(state_map)
    grid_state_map = np.reshape(state_map, (32, 32))
    normalized_grid = (grid_state_map - grid_state_map.mean()) / (grid_state_map.max() - grid_state_map.min())

    plt.imshow(normalized_grid, cmap='Blues', interpolation='nearest')
    plt.show()


def save_maze(v):
    np.save('maze_progress.npy', v)


def load_maze():
    progress = np.load('maze_progress.npy').item()
    print(progress)


#################################################################################


# creates random policy function
def create_random_policy(numberPossibleActions):
    
    A = np.ones(numberPossibleActions, dtype=float) / numberPossibleActions
    def policy_fn(observation):
        return A

    return policy_fn



# creates a greedy policy
def create_greedy_policy(Q):
    
    def policy_fn(state):

        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0

        return A

    return policy_fn




# Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
#    Finds an optimal greedy policy.
def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    
    
    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)
        
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        #if i_episode % 1000 == 0:
        #    print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        #    sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(1024):

            # Sample an action from our policy
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))

            if done: break
            state = next_state
        
        # Sum of discounted returns
        G = 0.0

        # The importance sampling ratio (the weights of the returns)
        W = 1.0
        
        # For each step in the episode, backwards
        for t in range(len(episode))[::-1]:
        
            state, action, reward = episode[t]
        
            # Update the total reward since step t
            G = discount_factor * G + reward
        
            # Update weighted importance sampling formula denominator
            C[state][action] += W
        
            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
        
            # If the action taken by the behavior policy is not the action 
            # taken by the target policy the probability will be 0 and we can break
            if action !=  np.argmax(target_policy(state)):  break
            W = W * 1./behavior_policy(state)[action]
        
    return Q, target_policy


random_policy = create_random_policy(env.numberOfPossibleActions)
print(random_policy)
Q, policy = mc_control_importance_sampling(env, num_episodes=100000, behavior_policy=random_policy)




# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():

    action_value = np.max(action_values)
    V[state] = action_value


plot_maze(V)
save_maze(V)

V = load_maze()
print(V)