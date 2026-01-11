# baseline validation for infinite deck blackjack environment

from infinite_env import BlackjackInfiniteEnv

def run_infinite_validation():
    """
    Runs a single hand in the infinite deck environment.
    In this setting, one hand = one episode.
    """
    
    # 1. Initialize the environment
    # Since D = infinity, the probabilities are constant.
    env = BlackjackInfiniteEnv(seed=42)
    
    # reset() starts the hand and deals the first card
    obs = env.reset()
    
    # The state in the infinite env is simpler: (Total, Usable Ace)
    # We do NOT need "count" or "remaining cards" here.
    player_sum, usable_ace = obs
    
    print("="*50)
    print("STARTING INFINITE DECK HAND (EPISODE)")
    print(f"Initial State -> Sum: {player_sum}, Usable Ace: {usable_ace}")
    print("="*50 + "\n")

    done = False
    
    # In the Infinite Env, this loop runs for exactly one hand.
    while not done:
        
        # BASELINE STRATEGY: Hit if total < 17
        action = 1 if player_sum < 17 else 0
        
        action_name = "HIT" if action == 1 else "STICK"
        print(f"Sum: {player_sum} | Action: {action_name}")
        
        # Advance the game
        obs, reward, done, info = env.step(action)
        
        # Update local variables
        player_sum, usable_ace = obs
        
        if not done:
            print(f" >> Drew a card. New Sum: {player_sum}")
        else:
            # Hand is over
            if player_sum > 21:
                print(f" >> RESULT: BUSTED. Reward: 0")
            else:
                print(f" >> RESULT: STICK. Reward: {reward}")

    print("\n" + "="*50)
    print(f"HAND FINISHED. Final Score: {reward}")
    print("="*50)

if __name__ == "__main__":
    run_infinite_validation()