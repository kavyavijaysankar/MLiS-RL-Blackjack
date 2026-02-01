
from finite_env import BlackjackFiniteEnv
import matplotlib.pyplot as plt

def baseline(num_decks=1, num_episodes=50000, seed=43):
    env = BlackjackFiniteEnv(num_decks=num_decks, seed=seed)
    total_rewards_all_episodes = []
    
    for i in range(num_episodes):
        obs = env.reset()
        total, usable_ace, tc_bucket, decks_bucket = obs
        
        done = False
        episode_reward = 0
        
        while not done:
            # Hit if under 17, else Stick
            action = 1 if total < 17 else 0 
            
            obs, reward, done, info = env.step(action)
            total, usable_ace, tc_bucket, decks_bucket = obs
            
            if info["hand_ended"]:
                episode_reward += reward

        total_rewards_all_episodes.append(episode_reward)
    
    # stats
    avg_score = sum(total_rewards_all_episodes) / num_episodes
    max_score = max(total_rewards_all_episodes)
    min_score = min(total_rewards_all_episodes)

    print(f"Total Episodes: {num_episodes}")
    print(f"Number of Decks: {num_decks}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Max Episode Score: {max_score}")
    print(f"Min Episode Score: {min_score}")
    
    return avg_score

if __name__ == "__main__":
    baseline()
