# Reinforcement Learning - Blackjack

This repository contains a reinforcement learning implementation designed to solve a stylised version of Blackjack. Developed as part of the Machine Learning in Science module at the University of Nottingham, the project evaluates the performance of Tabular Q-Learning agents across 2 environments:
1. **Infinite-Deck Blackjack Environment**: A stationary MDP where card draws are independent and identically distributed (i.i.d)
2. **Finite-Deck Blackjack Environment**: A non-stationary environment where sampling without replacement introduces temporal dependencies.

The agent does not play against a dealer. The goal is simply to maximize the hand total without exceeding 21. The quadratic reward function is given by:
$S = (\sum C_i)^2$ for totals $\leq 21$, and $0$ otherwise.

## Repository Structure
`agent.py`: Contains the BlackjackAgent (infinite) and Agent_finite_deck (finite) classes.

`finite_env.py` / `infinite_env.py`: Environments for the two deck types.

`training_finite.ipynb` / `training_infinite.ipynb`: Jupyter notebooks for model training and hyperparameter tuning.

`baseline_finite.py` / `baseline_infinite.py`: Scripts to evaluate the heuristic _Stand on 17_ policy.



## Design Choices

### Actions
The agent can choose between two actions:

- `HIT (1)`   → draw a new card  
- `STICK (0)` → stop drawing cards and end the hand  

### State Representation

The state is represented as a tuple: (player_sum, usable_ace)

- **player_sum**: the current total value of the player’s hand  
- **usable_ace**: `True` if the hand contains an Ace that can be counted as 11 without busting  

Additionally, to approximate a Markovian state in the non-stationary finite deck environment, the state is augmented with features tracking deck depletion:
- **True_count**: The running count normalized by the remaining decks, mapped to 3 bins. This tracks the statistical likelihood of high-value cards remaining in the deck(s).
- **Deck Depth**: The percentage of the deck already dealt, mapped to 3 bins. This indicates the reliability of the True Count as the deck(s) near depletion.

### Reward Function

Rewards are given **when a hand ends (for the infinite environment) or when an episode ends (for the finite environment)**:

- If `player_sum ≤ 21`:  reward = player_sum $^2$
- If `player_sum > 21` (bust): reward = 0

This reward structure encourages the agent to reach high but safe hand totals.

---

## Usage
To set up the environment and run the simulations, ensure you have Python 3.8+ installed. It is recommended that you use a virtual enviornment.

### Install the dependencies:
```
pip install numpy matplotlib scipy notebook
```
### Run the baselines
To establish the benchmark performance of the stationary "Stand on 17" heuristic, run the baseline scripts `baseline_finite.py` and `baseline_infinite.py`.

### Training and Evaluating the Agents
The Q-learning agents are implemented and trained within Jupyter Notebooks. These notebooks contain the training loops, hyperparameter configurations, and evaluation plots (Learning Curves and Score Distributions).

You can modify hyperparameters such as `learning_rate`, `discount_factor`, and `epsilon_decay` directly within the Agent class initialization in the notebooks to observe different learning dynamics. 
