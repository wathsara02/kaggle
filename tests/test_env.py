from omi_env.env import OmiEnv
from omi_env import rules


def test_env_seed_determinism():
    env_a = OmiEnv(seed=7)
    env_b = OmiEnv(seed=7)
    env_a.reset()
    env_b.reset()
    assert env_a.hands == env_b.hands


def test_env_trick_winner_info():
    env = OmiEnv(seed=1)
    env.reset()
    env.trump_suit = "S"
    env.stage = "play"
    env._init_selector(start=0)
    env.hands = [
        [rules.card_to_index(rules.Card("H", "8"))],
        [rules.card_to_index(rules.Card("H", "Q"))],
        [rules.card_to_index(rules.Card("S", "7"))],
        [rules.card_to_index(rules.Card("D", "A"))],
    ]
    env.lead_suit = None

    env.agent_selection = "player_0"
    env.step(env.hands[0][0])  # player 0 plays H8
    env.step(env.hands[1][0])  # player 1 plays HQ
    env.step(env.hands[2][0])  # player 2 plays S7 trump
    env.step(env.hands[3][0])  # player 3 plays DA completes trick

    # After one trick, teams should reflect winner (player 2 -> team 0)
    assert env.tricks_won[0] == 1
    assert env.tricks_won[1] == 0


def test_reward_shaping_logic():
    # Test illegal action penalty
    env = OmiEnv(seed=1, reward_shaping=True)
    env.reset()
    env._init_selector(start=0)
    # Player 0 hand has 4 cards after deal_first_four. Let's try an illegal trump action in play stage (if we were in play)
    # Actually, let's just use the existing mask logic.
    env.stage = "play"
    env.lead_suit = "H"
    env.hands[0] = [rules.card_to_index(rules.Card("H", "7"))]
    # Mask will only allow H7. Try playing something else.
    illegal_action = rules.card_to_index(rules.Card("S", "A")) 
    env.step(illegal_action)
    # p0 is team 0. Check reward.
    assert env._cumulative_rewards["player_0"] < 0  # Should have -0.1 penalty

    # Test trick reward
    env = OmiEnv(seed=1, reward_shaping=True)
    env.reset()
    env.trump_suit = "S"
    env.stage = "play"
    # Provide multiple cards so it's not terminal after one trick
    env.hands = [
        [rules.card_to_index(rules.Card("H", "8")), rules.card_to_index(rules.Card("D", "7"))],
        [rules.card_to_index(rules.Card("H", "7")), rules.card_to_index(rules.Card("D", "8"))],
        [rules.card_to_index(rules.Card("H", "A")), rules.card_to_index(rules.Card("D", "9"))],
        [rules.card_to_index(rules.Card("H", "9")), rules.card_to_index(rules.Card("D", "10"))],
    ]
    env._init_selector(start=0)
    env.step(env.hands[0][0]) # p0
    env.step(env.hands[1][0]) # p1
    env.step(env.hands[2][0]) # p2
    env.step(env.hands[3][0]) # p3 -> p2 wins trick
    # p2 is team 0. p0 and p2 should get +0.1 in their cumulative rewards
    assert env._cumulative_rewards["player_0"] == 0.1
    # For player_2: -0.05 (overplay) + 0.1 (trick) = 0.05
    assert env._cumulative_rewards["player_2"] == 0.05
    assert env._cumulative_rewards["player_1"] == 0.0

    # Test overplay penalty with choice
    env = OmiEnv(seed=1, reward_shaping=True)
    env.reset()
    env.stage = "play"
    env.lead_suit = "H"
    # p0 plays H8. p1 is teammate. p1 has HA (wins) and H7 (doesn't win).
    # If p1 plays HA, it's an overplay because p0 was already winning H8? 
    # WAIT, p0 played H8. If p1 plays HA, p1 is winning. p0 was winning before that.
    # So p1 took the trick from teammate p0.
    env.hands[0] = [rules.card_to_index(rules.Card("H", "8"))]
    env.hands[1] = [rules.card_to_index(rules.Card("H", "7")), rules.card_to_index(rules.Card("H", "A"))]
    env._init_selector(start=0)
    env.step(env.hands[0][0]) # p0 plays H8. Team 0 is winning.
    env.step(env.hands[1][1]) # p1 plays HA. Team 0 is still winning, but p1 took it from p0.
    # p1 is team 1. p0 is team 0. Wait, p1 and p3 are teammates. p0 and p2 are teammates.
    # p1 (team 1) took trick from p0 (team 0). That's NOT an overplay. That's a good move!
    
    # Correct Overplay test:
    # p0 plays H8. p1 plays H7. p2 (p0's teammate) plays HA.
    # p0 was winning with H8. p2 has H9 (still wins but lower) or H7 (doesn't win).
    # If p2 plays HA when he could have played something else to keep p0 winning? 
    # No, p2 just needs to NOT take it from p0 if p0 is already winning.
    env = OmiEnv(seed=1, reward_shaping=True)
    env.reset()
    env.stage = "play"
    env.lead_suit = "H"
    env.hands = [
        [rules.card_to_index(rules.Card("H", "J"))], # p0
        [rules.card_to_index(rules.Card("H", "7"))], # p1
        [rules.card_to_index(rules.Card("H", "8")), rules.card_to_index(rules.Card("H", "A"))], # p2 (teammate)
        [rules.card_to_index(rules.Card("H", "9"))], # p3
    ]
    env._init_selector(start=0)
    env.step(env.hands[0][0]) # p0 plays HJ. Team 0 (p0) is winning.
    env.step(env.hands[1][0]) # p1 plays H7. Team 0 (p0) is still winning.
    # Now p2's turn. HJ (11) is winning. p2 has H8 (8) and HA (14).
    # H8 is a "safe" move because HJ still wins.
    # If p2 plays HA, it's an OVERPLAY.
    env.step(env.hands[2][1]) # p2 plays HA.
    assert env._cumulative_rewards["player_2"] == -0.05
