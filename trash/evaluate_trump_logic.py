import torch
import numpy as np
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, get_device, build_policy
from omi_env.env import OmiEnv
from models.critic import encode_central_state

def evaluate_trump_picking(num_episodes=2):
    cfg = load_config("configs/lstm.yaml")
    device = get_device(True)
    
    # Load policy
    policy, _, __ = build_policy(cfg, device)
    weight_path = "runs/lstm_cpu/policy_last.pt"
    policy.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    policy.eval()

    env = OmiEnv(seed=42)
    
    total_trump_picks = 0
    optimal_trump_picks = 0
    random_matches = 0
    
    # Suit mapping
    suits = ['Spades', 'Hearts', 'Clubs', 'Diamonds']
    
    for ep in range(num_episodes):
        env.reset()
        
        # We need hidden states since it's an LSTM
        hidden_states = {
            agent: policy.init_hidden(1, device)
            for agent in env.agents
        }
        
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue
                
            obs_array = observation["observation"]
            hist_array = observation["history"]
            mask_array = observation["action_mask"]
            
            # Convert to tensors
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)
            hist_tensor = torch.tensor(hist_array, dtype=torch.float32).unsqueeze(0).to(device)
            mask_tensor = torch.tensor(mask_array, dtype=torch.bool).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, hidden_states[agent] = policy(
                    obs_tensor, hist_tensor,
                    hidden_states[agent],
                    action_mask=mask_tensor
                )
                
                # Deterministic or stochastic selection
                from scripts.eval import select_action
                action = select_action(policy, obs_tensor, hist_tensor, mask_tensor, device, deterministic=True)
                
            if env.stage == "trump":
                # Extract the 4 cards the declarer holds
                player_id = int(agent.split('_')[1])
                hand = env.hands[player_id]
                
                # The optimal choice is the suit they hold the most of
                # If there's a tie, any of the tied suits is considered optimal
                suit_counts = Counter(suits[c // 8] for c in hand)
                max_count = max(suit_counts.values()) if suit_counts else 0
                optimal_suits = [suit for suit, count in suit_counts.items() if count == max_count]
                
                chosen_suit_idx = action - 32
                chosen_suit = suits[chosen_suit_idx]
                
                total_trump_picks += 1
                if chosen_suit in optimal_suits:
                    optimal_trump_picks += 1
                    
                random_matches += (len(optimal_suits) / 4.0)
                
            env.step(action)
            
    with open("trump_eval.txt", "w") as f:
        f.write(f"--- Trump Selection Evaluation ---\n")
        f.write(f"Total Trump Declarations: {total_trump_picks}\n")
        f.write(f"AI chose the optimal (longest) suit: {optimal_trump_picks}/{total_trump_picks} ({(optimal_trump_picks/total_trump_picks)*100:.1f}%)\n")
        f.write(f"Random choice baseline expected: ~{int(random_matches)}/{total_trump_picks} ({(random_matches/total_trump_picks)*100:.1f}%)\n")
    
if __name__ == "__main__":
    evaluate_trump_picking()
