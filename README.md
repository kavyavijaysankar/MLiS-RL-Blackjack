# MLiS-RL-Blackjack

# Reinforcement Learning Blackjack Environments

This repository contains two Blackjack environments designed for Reinforcement Learning (RL) experiments:

1. **Infinite-Deck Blackjack Environment**
2. **Finite-Deck Blackjack Environment**

Both environments follow the standard RL interaction loop:
**Agent → Action → Environment → (State, Reward, Done)**

---

## Common Design Choices

### Actions
The agent can choose between two actions:

- `HIT (1)`   → draw a new card  
- `STICK (0)` → stop drawing cards and end the hand  

---

### State Representation

The state is represented as a tuple: (player_sum, usable_ace)

- **player_sum**: the current total value of the player’s hand  
- **usable_ace**: `True` if the hand contains an Ace that can be counted as 11 without busting  

This compact state captures all relevant information needed for optimal decision-making.

---

### Reward Function

Rewards are given **only when a hand ends**:

- If `player_sum ≤ 21`:  reward= player_sum^2
- If `player_sum > 21` (bust): reward=0

This reward structure encourages the agent to reach high but safe hand totals.

---

### Hand Value Calculation

Hand values are computed according to standard Blackjack rules:

1. All cards are first summed with Ace counted as `1`
2. If there is at least one Ace and `total + 10 ≤ 21`, one Ace is counted as `11`
3. The function returns:
 - the best achievable total
 - whether a usable Ace exists

---

## 1. Infinite-Deck Blackjack Environment

**Class:** `BlackjackInfiniteEnv`

### Key Idea
- There is **no stored deck**
- Each card draw is **independent and identically distributed**
- Card probabilities never change

This assumption simplifies the environment and makes it ideal for:
- Dynamic Programming
- Value Iteration
- Policy Iteration
- Theoretical RL analysis

### Card Distribution
Cards are drawn uniformly from: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
Where:
- Ace = 1  
- 2–9 = face value  
- 10/J/Q/K = 10  

### Episode Definition
- One episode corresponds to **one Blackjack hand**
- The episode ends when:
  - the agent busts, or
  - the agent chooses `STICK`

### Environment Dynamics
- `reset()` starts a new hand with one initial card
- `step(action)` advances the game by exactly one action
- After `done = True`, the agent must call `reset()` to start again

---

## 2. Finite-Deck Blackjack Environment

**Class:** `BlackjackFiniteEnv`

### Key Idea
- A **realistic finite deck** is explicitly stored
- Cards are **removed from the deck** when drawn
- The deck is shuffled at the start of an episode

### Deck Construction
- One suit contains: Where:
- Ace = 1  
- 2–9 = face value  
- 10/J/Q/K = 10  

### Episode Definition
- One episode corresponds to **one Blackjack hand**
- The episode ends when:
  - the agent busts, or
  - the agent chooses `STICK`

### Environment Dynamics
- `reset()` starts a new hand with one initial card
- `step(action)` advances the game by exactly one action
- After `done = True`, the agent must call `reset()` to start again

---

## 2. Finite-Deck Blackjack Environment

**Class:** `BlackjackFiniteEnv`

### Key Idea
- A **realistic finite deck** is explicitly stored
- Cards are **removed from the deck** when drawn
- The deck is shuffled at the start of an episode

### Deck Construction
- One suit contains:
- One deck contains **4 suits → 52 cards**
- Multiple decks are supported: total_cards = 52 × num_decks

- ### Episode Definition
- Each episode starts with a freshly built and shuffled deck
- Cards are drawn using `deck.pop()`
- The hand ends when:
- the agent busts, or
- the agent chooses `STICK`

This environment captures **card-depletion effects**, making it more realistic than the infinite-deck version.

---

## Differences Between Infinite and Finite Environments

| Feature | Infinite Deck | Finite Deck |
|------|--------------|-------------|
| Stored deck | ❌ No | ✅ Yes |
| Card depletion | ❌ No | ✅ Yes |
| Card probabilities | Fixed | Change over time |
| Complexity | Simpler | More realistic |
| Suitable for DP | Very suitable | More challenging |

---

## Intended Use

These environments are designed to be used with:
- Dynamic Programming agents
- Policy Iteration
- Value Iteration
- Monte Carlo methods

They provide a clean and modular foundation for studying decision-making under uncertainty in Blackjack.


