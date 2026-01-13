import random
from typing import List, Tuple, Optional, Dict, Any

# ACTION SPACE

# In Reinforcement Learning, the agent chooses from these discrete actions.
STICK = 0  # End the current hand and receive a reward based on the current sum.
HIT = 1    # Request another card to increase the hand sum.

def score(total: int) -> int:
    """
    Implements the Quadratic Reward Function: reward = total^2.
    
    As per the project description (Eq. 1), if the player stays under or 
    at 21, they receive the square of their total. If they exceed 21, 
    the reward for that hand is 0.
    """
    return total * total if total <= 21 else 0

def hand_value(cards: List[int]) -> Tuple[int, bool]:
    """
    Calculates the best possible hand value and checks for a "Usable Ace".
    
    Terminology:
    - Ace is represented as 1 in the deck.
    - An Ace is "usable" if it can be counted as 11 without the total exceeding 21.
    - Logic: Initially count all Aces as 1. If an Ace can be converted to 11 
      (+10 points) safely, we do so and flag it as 'usable'.
    """
    total = sum(cards) 
    usable_ace = False
    
    # Check if converting one Ace from 1 to 11 is beneficial and safe
    if 1 in cards and total + 10 <= 21:
        total += 10
        usable_ace = True
    return total, usable_ace

class BlackjackFiniteEnv:
    """
    A Finite Deck (D < ∞) Blackjack Environment.
    
    Structure:
    - Episode: Lasts until the entire shoe (D decks) is empty.
    - Hand: A sub-cycle within an episode. Rewards are given at the end of hands.
    - State: Includes 'Memory' (Card Counting) to help the agent make informed 
      decisions as the deck composition changes over time.
    """

    def __init__(self, num_decks: int = 1, seed: Optional[int] = None):
        """
        Initializes the environment.
        :param num_decks: Number of standard 52-card decks in the shoe.
        :param seed: Random seed for reproducibility of shuffles and draws.
        """
        self.rng = random.Random(seed)
        self.num_decks = num_decks
        
        # Standard deck construction:
        # Values 1(Ace) through 9 appear 4 times each. 
        # 10, J, Q, K all carry a value of 10, so 10 appears 16 times per deck.
        self.standard_deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        
        self.deck: List[int] = []          # The current physical cards in the shoe.
        self.player_cards: List[int] = []  # Cards currently held by the player in the hand.
        self.running_count = 0             # The Hi-Lo card counting memory component.
        
        self.reset_deck()

    def reset_deck(self):
        """
        Re-initializes and shuffles the full shoe of cards.
        This is called only at the start of a new episode or when the deck is empty.
        The running count is reset to 0 as the 'memory' starts fresh.
        """
        self.deck = self.standard_deck * self.num_decks
        self.rng.shuffle(self.deck)
        self.running_count = 0 

    def draw_card(self) -> Optional[int]:
        """
        Draws the top card from the deck and updates the 'Running Count'.
        
        Memory Logic (Hi-Lo System):
        - High Cards (10s and Aces) are bad for the deck's future (count -1).
        - Low Cards (2-6) are good for the deck's future (count +1).
        - Neutral Cards (7-9) do not change the count.
        """
        if not self.deck:
            return None
        
        card = self.deck.pop()
        
        # Update memory: subtract for high cards, add for low cards.
        if card in [1, 10]: 
            self.running_count -= 1
        elif card in [2, 3, 4, 5, 6]:
            self.running_count += 1
            
        return card

    def get_state(self) -> Tuple[int, bool, int, int]:
        """
        Defines the 'Observation' sent to the RL Agent.
        
        The state must contain enough information for the agent to learn:
        1. Current hand total (to know if it should hit/stick).
        2. Usable Ace status (changes the risk of hitting).
        3. Running Count (memory of past cards).
        4. Cards Remaining (to know the 'weight' or reliability of the count).
        """
        total, usable_ace = hand_value(self.player_cards)
        return (total, usable_ace, self.running_count, len(self.deck))

    def reset(self) -> Tuple[int, bool, int, int]:
        """
        Starts a brand new episode. Shuffles the deck and deals the 
        first card of the first hand.
        """
        self.reset_deck()
        self.player_cards = [self.draw_card()]
        return self.get_state()

    def step(self, action: int) -> Tuple[Tuple[int, bool, int, int], int, bool, Dict[str, Any]]:
        """
        Processes the player's action and advances the environment.
        
        Returns:
            - state: The new observation.
            - reward: The quadratic score if a hand ended, else 0.
            - done: True if the deck is empty (End of Episode).
            - info: Diagnostic data (bust status, hand transitions).
        """
        reward = 0
        done = False
        info = {"hand_ended": False}

        if action == HIT:
            card = self.draw_card()
            if card is not None:
                self.player_cards.append(card)
                total, _ = hand_value(self.player_cards)
                if total > 21: # Player Busts
                    info["hand_ended"] = True
                    info["bust"] = True
                    # Reward is 0 for a bust, which is the default.
            else:
                done = True # Deck ran out while player was hitting.

        elif action == STICK:
            # Calculate the reward based on the final total of the hand.
            total, _ = hand_value(self.player_cards)
            reward = score(total) 
            info["hand_ended"] = True
            info["bust"] = False

        # TRANSITION LOGIC: 
        # If the hand just ended but cards remain in the deck, start the next hand.
        if info["hand_ended"] and not done:
            if len(self.deck) > 0:
                self.player_cards = [self.draw_card()]
            else:
                # No cards left to start a new hand; the episode is officially over.
                done = True 
        
        return self.get_state(), reward, done, info