import torch
import numpy as np
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, get_device, build_policy
from omi_env.env import OmiEnv

def probe_trump(num_hands=100):
    cfg = load_config("configs/lstm.yaml")
    device = get_device(True)
    policy, _, __ = build_policy(cfg, device)
    policy.load_state_dict(torch.load("runs/lstm_cpu/policy_last.pt", map_location=device, weights_only=True))
    policy.eval()

    env = OmiEnv()
    suits = ['Spades', 'Hearts', 'Clubs', 'Diamonds']
    ai_suit_lengths = []
    max_suit_lengths = []
    random_suit_lengths = []
    
    for i in range(num_hands):
        env.reset(seed=1000+i)
        agent = env.agent_selection
        obs = env.observe(agent)
        
        obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0).to(device)
        hist_tensor = torch.tensor(obs["history"], dtype=torch.float32).unsqueeze(0).to(device)
        mask_tensor = torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0).to(device)
        
        with torch.no_grad():
            hidden = policy.init_hidden(1, device)
            logits, _ = policy(obs_tensor, hist_tensor, hidden, action_mask=mask_tensor)
            action = torch.argmax(logits, dim=-1).item()
            
        player_id = int(agent.split('_')[1])
        hand = env.hands[player_id]
        suit_counts = Counter(suits[c // 8] for c in hand)
        
        chosen_suit = suits[action-32]
        ai_suit_lengths.append(suit_counts.get(chosen_suit, 0))
        max_suit_lengths.append(max(suit_counts.values()))
        random_suit_lengths.append(np.mean(list(suit_counts.values())) if suit_counts else 0)

    with open("trump_audit.txt", "w") as f:
        f.write(f"--- Trump Selection Audit (100 Hands) ---\n")
        f.write(f"Avg cards in AI-chosen suit: {np.mean(ai_suit_lengths):.2f}\n")
        f.write(f"Avg cards in Max-length suit: {np.mean(max_suit_lengths):.2f}\n")
        f.write(f"Avg cards in Random-chosen suit: {np.mean(random_suit_lengths):.2f}\n")
        f.write(f"Efficiency (AI length / Optimal length): { (np.mean(ai_suit_lengths) / np.mean(max_suit_lengths))*100:.1f}%\n")

if __name__ == "__main__":
    probe_trump()
