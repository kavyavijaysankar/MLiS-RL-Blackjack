import numpy as np
from infinite_env import BlackjackInfiniteEnv

def run_baseline_validation(episodes_to_average=100000):
    # 1. Single Hand Demonstration
    env = BlackjackInfiniteEnv(seed=42)
    obs = env.reset()
    player_sum, usable_ace = obs
    
    print("="*50)
    print("SINGLE HAND BASELINE DEMONSTRATION")
    print(f"Initial State -> Sum: {player_sum}, Usable Ace: {usable_ace}")
    print("="*50 + "\n")

    done = False
    while not done:
        # BASELINE STRATEGY: Hit if total < 17
        action = 1 if player_sum < 17 else 0
        action_name = "HIT" if action == 1 else "STICK"
        print(f"Sum: {player_sum} | Action: {action_name}")
        
        obs, reward, done, _ = env.step(action)
        player_sum, usable_ace = obs
        
        if not done:
            print(f" >> Drew a card. New Sum: {player_sum}")
        else:
            if player_sum > 21:
                print(f" >> RESULT: BUSTED. Reward: 0")
            else:
                print(f" >> RESULT: STICK. Reward: {reward}")

    # 2. Print Baseline Strategy Table
    print("\n" + "="*45)
    print("FIXED BASELINE STRATEGY")
    print("="*45)
    print(f"{'Sum':<6} | {'Action':<18}")
    print("-" * 45)

    for s in range(11, 22):
        decision = "HIT" if s < 17 else "STICK"
        print(f" {s:<5} | {decision:<18}")
    print("="*45)

if __name__ == "__main__":
    run_baseline_validation()