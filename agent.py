import random
import numpy as np
from collections import defaultdict

class BlackjackAgent:
    """
    Reinforcement Learning Agent using Q-Learning to play Blackjack.
    
    The agent maintains a 'Q-Table' (a dictionary) where it stores the 
    estimated value of taking an action (Hit or Stick) in a specific state.
    """
    
    def __init__(self, alpha=0.05, gamma=0.9, epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.9995):
        """
        Initialize the agent with Hyperparameters.
        
        :param alpha: Learning Rate (how much new info overrides old info).
        :param gamma: Discount Factor (how much we value future rewards).
        :param epsilon: Starting exploration rate (1.0 = 100% random).
        :param min_epsilon: The lowest we want epsilon to go.
        :param epsilon_decay: How fast epsilon shrinks as the agent learns.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        # The Q-Table: Maps State -> [Value of STICK, Value of HIT]
        # We use a defaultdict so that any new state is initialized to [0.0, 0.0]
        self.q_table = defaultdict(lambda: [0.0, 0.0])

    def get_action(self, state):
        """
        Choose an action using the Epsilon-Greedy Policy.
        
        1. With probability epsilon, choose a RANDOM action (Exploration).
        2. Otherwise, choose the action with the HIGHEST Q-value (Exploitation).
        """
        if random.random() < self.epsilon:
            return random.choice([0, 1])  # 0=STICK, 1=HIT
        
        # Get the index of the highest value (0 or 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-Learning formula.
        
        Formula: Q(s,a) = Q(s,a) + alpha * [Reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        # Current predicted value for this action
        old_value = self.q_table[state][action]
        
        # Estimate of the best future reward from the next state
        if done:
            next_max = 0  # No future rewards if the episode is over
        else:
            next_max = np.max(self.q_table[next_state])
        
        # The new estimate: Immediate reward + discounted future reward
        new_estimate = reward + (self.gamma * next_max)
        
        # Update the Q-value for the current state-action pair
        self.q_table[state][action] += self.alpha * (new_estimate - old_value)

    def decay_epsilon(self):
        """
        Reduces epsilon over time. This makes the agent start by 
        exploring randomly and finish by playing its best learned strategy.
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def get_q_value(self, state, action):
        """Helper to look up a value for debugging."""
        return self.q_table[state][action]