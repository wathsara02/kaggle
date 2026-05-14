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

    # Player 2 wins with trump, so team 0 scores.
    assert env.tricks_won[0] == 1
    assert env.tricks_won[1] == 0


def test_reward_shaping_logic():
    env = OmiEnv(seed=1, reward_shaping=True)
    env.reset()
    env._init_selector(start=0)
    env.stage = "play"
    env.lead_suit = "H"
    env.hands[0] = [rules.card_to_index(rules.Card("H", "7"))]
    illegal_action = rules.card_to_index(rules.Card("S", "A")) 
    env.step(illegal_action)
    assert env._cumulative_rewards["player_0"] < 0

    env = OmiEnv(seed=1, reward_shaping=True)
    env.reset()
    env.trump_suit = "S"
    env.stage = "play"
    # Keep one card after the trick so terminal reward does not apply.
    env.hands = [
        [rules.card_to_index(rules.Card("H", "8")), rules.card_to_index(rules.Card("D", "7"))],
        [rules.card_to_index(rules.Card("H", "7")), rules.card_to_index(rules.Card("D", "8"))],
        [rules.card_to_index(rules.Card("H", "A")), rules.card_to_index(rules.Card("D", "9"))],
        [rules.card_to_index(rules.Card("H", "9")), rules.card_to_index(rules.Card("D", "10"))],
    ]
    env._init_selector(start=0)
    env.step(env.hands[0][0])
    env.step(env.hands[1][0])
    env.step(env.hands[2][0])
    env.step(env.hands[3][0])
    assert env._cumulative_rewards["player_0"] == 0.1
    assert env._cumulative_rewards["player_2"] == 0.05
    assert env._cumulative_rewards["player_1"] == 0.0

    # Opponent taking the trick is not overplay.
    env = OmiEnv(seed=1, reward_shaping=True)
    env.reset()
    env.stage = "play"
    env.lead_suit = "H"
    env.hands[0] = [rules.card_to_index(rules.Card("H", "8"))]
    env.hands[1] = [rules.card_to_index(rules.Card("H", "7")), rules.card_to_index(rules.Card("H", "A"))]
    env._init_selector(start=0)
    env.step(env.hands[0][0])
    env.step(env.hands[1][1])
    
    # Teammate overplays while a safe card is available.
    env = OmiEnv(seed=1, reward_shaping=True)
    env.reset()
    env.stage = "play"
    env.lead_suit = "H"
    env.hands = [
        [rules.card_to_index(rules.Card("H", "J"))],
        [rules.card_to_index(rules.Card("H", "7"))],
        [rules.card_to_index(rules.Card("H", "8")), rules.card_to_index(rules.Card("H", "A"))],
        [rules.card_to_index(rules.Card("H", "9"))],
    ]
    env._init_selector(start=0)
    env.step(env.hands[0][0])
    env.step(env.hands[1][0])
    env.step(env.hands[2][1])
    assert env._cumulative_rewards["player_2"] == -0.05
