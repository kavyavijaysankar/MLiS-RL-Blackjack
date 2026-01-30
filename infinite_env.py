import random
from typing import List, Tuple, Optional, Dict, Any

# Actions
STICK = 0 # stop
HIT = 1 # draw a card

# State: (player_sum, usable_ace)
State = Tuple[int, bool]

def score(total: int) -> int:
  
    return total * total if total <= 21 else 0


def hand_value(cards: List[int]) -> State:
    """
According to the hand list:
- player_sum: The most advantageous total according to Blackjack rules
- usable_ace: Is there an Ace that can be counted as 11?
Card representation:
- Ace = 1
- 2-9 = itself
- 10/J/Q/K = 10

Logic:
1) First, add them all normally (Ace=1)
2) If there is an Ace and total+10 <= 21, count an Ace as 11 (i.e., add +10)
"""
    total = sum(cards)
    usable_ace = False

    # if ace
    if 1 in cards and total + 10 <= 21:
        total += 10
        usable_ace = True

    return total, usable_ace

# 3) Environment 
class BlackjackInfiniteEnv:

# Infinite deck Blackjack environment.
# No deck list. new card drawn randomly with each hit
# 1 Episode = 1 hand. Done = True when the hand is finished.

    def __init__(self, seed: Optional[int] = None):
    
        self.rng = random.Random(seed)   
        self.player_cards: List[int] = []  
        self.done: bool = False           

    # draw card
    def draw_card(self) -> int:
        """
      model:
          - Ace: 1
          - 2..9
          - 10 (10/J/Q/K all 10)
        """
        return self.rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9,10,10,10,10])

    # State
    def get_state(self) -> State:
        """
        Agent will see as:
          (player_sum, usable_ace)
        """
        return hand_value(self.player_cards)

    # Reset
    def reset(self) -> State:
        
        self.player_cards = []
        self.done = False 

        first = self.draw_card()
        self.player_cards.append(first)

        return self.get_state()

    # Valid actions
    def valid_actions(self, state: State):

        if self.done:
            return [] #no action if deck is over
        return [STICK, HIT] #if not finished, hit or stick.

    # Step
    #agent's movement and action 1 step forward

    def step(self, action: int) -> Tuple[State, int, bool, Dict[str, Any]]:
        
        if self.done:
            return self.get_state(), 0, True, {"error": "Episode already done. Call reset()."}

        if action == HIT:
            card = self.draw_card()
            self.player_cards.append(card)

            total, usable = self.get_state()

            if total > 21:
                self.done = True
                return (total, usable), 0, True, {"hand_ended": True, "bust": True, "card": card}

            return (total, usable), 0, False, {"hand_ended": False, "card": card}

        elif action == STICK:
            total, usable = self.get_state()
            r = score(total)
            self.done = True
            return (total, usable), r, True, {"hand_ended": True, "bust": False}

        else:
            raise ValueError("Invalid action. Use HIT=1 or STICK=0.")

