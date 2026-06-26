import lightgbm as lgb
import random
from cg.api import Observation, to_observation_class
import os
from cg.game import battle_start, battle_select, battle_finish
from cg.api import all_card_data, all_attack, search_begin, search_step, search_end, search_release
from time import perf_counter as pf

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

    legal_moves = obs.select.option

    best_move = None
    highest_score = -float('inf')

    for move in [m.type for m in legal_moves]:
        your_index = obs.current.yourIndex
        state = obs.current
        active = state.players[1 - your_index].active
        search = search_begin(
            obs,
            your_deck=random.sample(deck_0, state.players[your_index].deckCount), # Randomly select from deck.
            your_prize=random.sample(deck_0, len(state.players[your_index].prize)), # Randomly select from deck.
            opponent_deck=[1072] * state.players[1 - your_index].deckCount, # Fill with Snorlax (There is no deep meaning).
            opponent_prize=[1] * len(state.players[1 - your_index].prize), # Fill with Basic Energy (There is no deep meaning)
            opponent_hand=[1] * state.players[1 - your_index].handCount, # Fill with Basic Energy.
            opponent_active=[1072] if len(active) > 0 and active[0] == None else []) # Fill with Snorlax.
        hypothetical_state = search_step(search.searchId, move)

        board_features = extract_features(hypothetical_state)

        score = bst.predict([board_features])[0]

        if score > highest_score:
            highest_score = score
            best_move = move

        search_end(search.searchId)
        search_release(search.searchId)
    print(f"Best move: {best_move} with score: {highest_score}")
    return best_move

def stop_watch(st, message):
    time = pf()-st
    print(f"{message} took {time:.4f} seconds")

st = pf()

bst = lgb.Booster(model_file='pok_model.txt')
deck_0_path = "deck.csv"
deck_1_path = "deck.csv"

stop_watch(st, "Model loaded")

deck_0 = read_deck_csv(deck_0_path)
deck_1 = read_deck_csv(deck_1_path)

stop_watch(st, "Decks loaded")

while True:
    obs, start_data = battle_start(deck_0, deck_1)
    if obs is None:
        print("Battle start failed.")
        break

    stop_watch(st, "Battle started")

    while True:
        move = agent_blueprint(obs)
        obs = battle_select([move])
        stop_watch(st, "Move selected")
        if obs is None:
            print("Battle finished.")
            break

    battle_finish()
