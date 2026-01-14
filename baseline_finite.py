# baseline finite deck episode to validate environment logic

from finite_env import BlackjackFiniteEnv

def run_validation_episode():
    """
    Runs a single full episode (until the deck is empty) to test 
    the environment logic and establish a baseline score.
    """
    
    # 1. Initialize the environment
    # We use 1 deck and a fixed seed so you can verify the math manually.
    env = BlackjackFiniteEnv(num_decks=1, seed=42)
    
    # Reset starts the EPISODE (shuffles everything)
    obs = env.reset()
    
    # Unpack the 4-part state: (Total, Usable Ace, Running Count, Cards Left)
    total, usable_ace, count, remaining = obs
    
    print("="*50)
    print("STARTING VALIDATION EPISODE (FINITE DECK)")
    print(f"Initial State -> Sum: {total}, Count: {count}, Deck: {remaining}")
    print("="*50 + "\n")

    done = False
    total_episode_reward = 0
    hand_count = 1
    
    # The Episode Loop: Continues until the deck is exhausted
    while not done:
        
        # BASELINE STRATEGY: 
        # We hit if the sum is less than 17. 
        # This doesn't use the 'count' yet—that's the RL agent's future job!
        action = 1 if total < 17 else 0
        
        action_name = "HIT" if action == 1 else "STICK"
        print(f"[Hand {hand_count}] Sum: {total} | Count: {count} | Action: {action_name}")
        
        # Advance the game
        obs, reward, done, info = env.step(action)
        
        # Update our tracking variables from the new state
        total, usable_ace, count, remaining = obs
        
        # Check if the hand finished (either by sticking or busting)
        if info["hand_ended"]:
            if info.get("bust"):
                print(f" >> RESULT: BUSTED. Reward: 0")
            else:
                # Reward is calculated as (total^2)
                print(f" >> RESULT: STICK. Reward: {reward}")
            
            total_episode_reward += reward
            
            # If the episode isn't over, prepare for the next hand
            if not done:
                print(f"--- NEW HAND DEALT | Cards left: {remaining} ---\n")
                hand_count += 1
            else:
                print(f"--- DECK EXHAUSTED ---\n")

    print("="*50)
    print("EPISODE SUMMARY")
    print(f"Total Hands Played: {hand_count}")
    print(f"Total Accumulated Score: {total_episode_reward}")
    print("="*50)

if __name__ == "__main__":
    run_validation_episode()