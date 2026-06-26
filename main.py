import lightgbm as lgb
import numpy as np
from cg.api import Observation, to_observation_class
import os
from cd.game import battle_start, battle_select, battle_finish
from cg.api import all_card_data, all_attack, search_begin, search_step, search_end, search_release

def read_deck_csv(file_path) -> list[int]:
    """Read deck.csv.
    
    Returns:
        list[int]: A list of card IDs in the deck.
    """
    
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck

def extract_features(state):
    features = []

    features.append(state.get('my_prize_cards'))
    features.append(state.get('opponent_prize_cards'))

    features.append(state.get('my_active_pokemon_hp'))
    features.append(state.get('opponent_active_pokemon_hp'))

    cards_in_hand = state.get('my_hand')
    energy_count = sum(1 for card in cards_in_hand if "Energy" in card.get("type",""))
    features.append(energy_count)
    
    return features

def agent_blueprint(obs_dict):

    obs: Observation = to_observation_class(obs_dict)

    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return read_deck_csv()

    legal_moves = obs.legal_moves

    best_move = None
    highest_score = -float('inf')

    for move_index in legal_moves:
        search = search_begin(obs)

        hypothetical_state = search_step(search.search_id, move_index)

        board_features = extract_features(hypothetical_state)

        score = bst.predict([board_features])[0]

        if score > highest_score:
            highest_score = score
            best_move = move_index

        search_end(search.search_id)
        search_release(search.search_id)

    return best_move

bst = lgb.Booster(model_file='pok_model.txt')
deck_0_path = "deck.csv"
deck_1_path = "deck.csv"

deck_0 = read_deck_csv(deck_0_path)
deck_1 = read_deck_csv(deck_1_path)

while True:
    obs, start_data = battle_start(deck_0, deck_1)
    if obs is None:
        print("Battle start failed.")
        break

    while True:
        move = agent_blueprint(obs)
        obs = battle_select([move])
        if obs is None:
            print("Battle finished.")
            break

    battle_finish()
