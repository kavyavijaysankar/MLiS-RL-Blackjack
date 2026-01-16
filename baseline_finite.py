# finite deck episode to validate environment logic & get baseline score

from finite_env import BlackjackFiniteEnv

# runs a full episode to get a baseline score
def run_validation_episode():
    env = BlackjackFiniteEnv(num_decks=1, seed=42) # set number of decks
    
    # start episode
    obs = env.reset()
    
    # Unpack the 4-part state: (Total, Usable Ace, Running Count, Cards Left)
    total, usable_ace, count, remaining = obs
    
    print(f"Initial State -> Sum: {total}, Count: {count}, Deck: {remaining}")

    done = False
    total_episode_reward = 0
    hand_count = 1
    
    # episode loop
    while not done:
        action = 1 if total < 17 else 0 # hit if under 17, else stick
        
        action_name = "HIT" if action == 1 else "STICK"
        print(f"[Hand {hand_count}] Sum: {total} | Count: {count} | Action: {action_name}")

        obs, reward, done, info = env.step(action)
        
        # Update our tracking variables from the new state
        total, usable_ace, count, remaining = obs
        
        # Check if the hand finished (either by sticking or busting)
        if info["hand_ended"]:
            if info.get("bust"):
                print(f"RESULT: BUSTED. Reward: 0")
            else:
                # Reward is calculated as (total^2)
                print(f"RESULT: STICK. Reward: {reward}")
            total_episode_reward += reward
            
            # next hand or end of episode
            if not done:
                print(f"NEW HAND DEALT | Cards left: {remaining}\n")
                hand_count += 1
            else:
                print(f"DECK EXHAUSTED\n")

    print("EPISODE SUMMARY")
    print(f"Total Hands Played: {hand_count}")
    print(f"Total Accumulated Score: {total_episode_reward}")



if __name__ == "__main__":
    run_validation_episode()