# MDP example implementation
import gym
import numpy as np

# Create the environment
env = gym.make('FrozenLake-v0')

# Set the discount factor
gamma = 0.9

# Initialize the state-value function
V = np.zeros(env.observation_space.n)

# Define the policy
def policy(state):
    return np.argmax(Q[state, :])

# Initialize the action-value function
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Perform value iteration
for i in range(1000):
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            # Compute the expected value of taking the given action
            next_states, rewards, _, _ = env.P[state][action]
            expected_value = np.sum([p * (r + gamma * V[s]) for p, r, s, _ in next_states])
            Q[state][action] = expected_value
    # Update the state-value function
    V = np.max(Q, axis=1)

# Print the optimal policy
print([env.action_names[policy(s)] for s in range(env.observation_space.n)])
