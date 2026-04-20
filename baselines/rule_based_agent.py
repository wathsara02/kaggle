import numpy as np

from omi_env import rules


class RuleBasedAgent:
    """
    Simple heuristic agent:
    - During trump declaration: pick the suit with the most cards in hand.
    - During play: follow suit with highest card when possible, otherwise play lowest card.
    """

    def act(self, observation: dict) -> int:
        mask = observation["action_mask"]
        obs_vec = observation["observation"]
        n = rules.NUM_CARDS
        hand_mask = obs_vec[:n]
        trump_onehot = obs_vec[n : n + 4]
        lead_onehot = obs_vec[n + 4 : n + 8]
        legal = np.nonzero(mask)[0]

        # Trump declaration stage
        if np.any(mask[rules.ACTION_TRUMP_OFFSET : rules.ACTION_TRUMP_OFFSET + 4]):
            counts = []
            for i, suit in enumerate(rules.SUITS):
                suit_indices = [idx for idx in range(n) if rules.index_to_card(idx).suit == suit]
                counts.append(hand_mask[suit_indices].sum())
            best_suit = int(np.argmax(counts))
            return rules.ACTION_TRUMP_OFFSET + best_suit

        # Card play stage
        lead_suit = None
        if lead_onehot.sum() > 0:
            lead_suit = rules.SUITS[int(np.argmax(lead_onehot))]

        legal_cards = [a for a in legal if a < n]
        if lead_suit:
            lead_cards = [c for c in legal_cards if rules.index_to_card(c).suit == lead_suit]
            if lead_cards:
                return max(lead_cards, key=lambda c: rules.index_to_card(c).value)
        # Otherwise play lowest ranked legal card to save high cards
        return min(legal_cards, key=lambda c: rules.index_to_card(c).value)
