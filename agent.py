import random
import numpy as np
from collections import defaultdict

class BlackjackAgent: # agent for infinite deck
    
    def __init__(self, alpha=0.05, gamma=0.9, epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.9995):
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        # The Q-Table: Maps State -> [Value of STICK, Value of HIT]
        # We use a defaultdict so that any new state is initialized to [0.0, 0.0]
        self.q_table = defaultdict(lambda: [0.0, 0.0])

    def get_action(self, state): # chooses with the epsilon greedy policy
        """
        With probability epsilon, choose a RANDOM action (Exploration).
        Otherwise, choose the action with the HIGHEST Q-value (Exploitation).
        """
        if random.random() < self.epsilon:
            return random.choice([0, 1])  # 0=stick, 1=hit
        
        # Get the index of the highest value (0 or 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):        
        # Q(s,a) = Q(s,a) + alpha * [Reward + gamma * max(Q(s',a')) - Q(s,a)]
        # Current predicted value for this action
        old_value = self.q_table[state][action]
        
        # Estimate of the best future reward from the next state
        if done:
            next_max = 0  # No future rewards if the episode is over
        else:
            next_max = np.max(self.q_table[next_state])
        # The new estimate: Immediate reward + discounted future reward
        new_estimate = reward + (self.gamma * next_max)
        # Update the Q-value
        self.q_table[state][action] += self.alpha * (new_estimate - old_value)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def get_q_value(self, state, action):
        return self.q_table[state][action]
    
# agent speficific to the finite deck
class Agent_finite_deck:
    def __init__(self, alpha=0.01, gamma=0.95, epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.99999):
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        # biased initiaization 
        self.q_table = defaultdict(lambda: [0.0, 500.0]) # encourages hitting in new states to learn more

    def get_action(self, state):
        # Choose an action using the Epsilon-Greedy Policy.
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        # Update the Q-table using the Q-Learning formula.
        old_val = self.q_table[state][action]
        if done:
            next_max = 0
        else:
            next_max = np.max(self.q_table[next_state])
        # Q(s,a) = Q(s,a) + alpha * [Reward + gamma * max(Q(s',a')) - Q(s,a)]
        new_val = old_val + self.alpha * (reward + self.gamma * next_max - old_val)
        self.q_table[state][action] = new_val

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def get_q_value(self, state, action):
        return self.q_table[state][action]

