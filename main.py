import lightgbm as lgb
import random
from cg.api import Observation, to_observation_class
import os
from cg.game import battle_start, battle_select, battle_finish
from cg.api import all_card_data, all_attack, search_begin, search_step, search_end, search_release
from time import perf_counter as pf
from datetime import datetime as dt

output_path = "\\games\\selfplay\\"

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

SCALAR_COLUMNS = [
    "file",
    "step_index",
    "entry_index",
    "status",
    "select_context",
    "select_type",
    "select_option_count",
    "yourIndex",
    "turn",
    "turnActionCount",
    "firstPlayer",
    "energyAttached",
    "supporterPlayed",
    "stadiumPlayed",
    "retreated",
    "result",
]

PLAYER_COLUMNS = [
    "deckCount",
    "discardCount",
    "prizeCount",
    "handCount",
    "benchCount",
    "activeExists",
    "activeHp",
    "activeMaxHp",
    "activeEnergyCount",
    "activeToolCount",
    "activeAppearThisTurn",
    "poisoned",
    "burned",
    "asleep",
    "paralyzed",
    "confused",
]

HEADER = SCALAR_COLUMNS + [f"p{player}_{col}" for player in (0, 1) for col in PLAYER_COLUMNS] + ["label"]


def to_int(value):
    if isinstance(value, bool):
        return 1 if value else 0
    if value is None:
        return 0
    return int(value)


def extract_active_features(active_list):
    if not active_list or active_list[0] is None:
        return {
            "activeExists": 0,
            "activeHp": 0,
            "activeMaxHp": 0,
            "activeEnergyCount": 0,
            "activeToolCount": 0,
            "activeAppearThisTurn": 0,
        }

    active = active_list[0]
    return {
        "activeExists": 1,
        "activeHp": to_int(active.get("hp", 0)),
        "activeMaxHp": to_int(active.get("maxHp", 0)),
        "activeEnergyCount": len(active.get("energies", [])) if isinstance(active.get("energies"), list) else 0,
        "activeToolCount": len(active.get("tools", [])) if isinstance(active.get("tools"), list) else 0,
        "activeAppearThisTurn": to_int(active.get("appearThisTurn", False)),
    }


def player_features(player):
    result = {
        "deckCount": to_int(player.get("deckCount", 0)),
        "discardCount": len(player.get("discard", [])) if isinstance(player.get("discard"), list) else 0,
        "prizeCount": len(player.get("prize", [])) if isinstance(player.get("prize"), list) else 0,
        "handCount": to_int(player.get("handCount", 0)),
        "benchCount": len(player.get("bench", [])) if isinstance(player.get("bench"), list) else 0,
        "poisoned": to_int(player.get("poisoned", False)),
        "burned": to_int(player.get("burned", False)),
        "asleep": to_int(player.get("asleep", False)),
        "paralyzed": to_int(player.get("paralyzed", False)),
        "confused": to_int(player.get("confused", False)),
    }
    result.update(extract_active_features(player.get("active", [])))
    return result


def extract_features(entry):
    global step_index
    step_index += 1

    obs = entry.get("observation")
    current = obs.get("current")
    if current is None:
        return None

    your_index = current.get("yourIndex")
    if your_index is None:
        return None


    row = {
        "file": start_data.battlePtr,
        "step_index": step_index,
        "entry_index": current.get("yourIndex"),
        "status": entry.get("status", ""),
        "select_context": obs.get("select", {}).get("context") if obs.get("select") else None,
        "select_type": obs.get("select", {}).get("type") if obs.get("select") else None,
        "select_option_count": len(obs.get("select", {}).get("option", [])) if obs.get("select") else 0,
        "yourIndex": to_int(your_index),
        "turn": to_int(current.get("turn", 0)),
        "turnActionCount": to_int(current.get("turnActionCount", 0)),
        "firstPlayer": to_int(current.get("firstPlayer", -1)),
        "energyAttached": to_int(current.get("energyAttached", False)),
        "supporterPlayed": to_int(current.get("supporterPlayed", False)),
        "stadiumPlayed": to_int(current.get("stadiumPlayed", False)),
        "retreated": to_int(current.get("retreated", False)),
        "result": to_int(current.get("result", -1)),
        # "label": 1 if label == 1 else 0,
    }

    players = current.get("players", [])
    if len(players) < 2:
        return None

    for player_index in (0, 1):
        player = players[player_index]
        prefix = f"p{player_index}_"
        player_feats = player_features(player)
        for key, value in player_feats.items():
            row[prefix + key] = value

    return row

# def extract_features(state):
#     features = []

#     features.append(state.get('my_prize_cards'))
#     features.append(state.get('opponent_prize_cards'))

#     features.append(state.get('my_active_pokemon_hp'))
#     features.append(state.get('opponent_active_pokemon_hp'))

#     cards_in_hand = state.get('my_hand')
#     energy_count = sum(1 for card in cards_in_hand if "Energy" in card.get("type",""))
#     features.append(energy_count)
    
#     return features

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

    for move in range(len(legal_moves)):
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
        hypothetical_state = search_step(search.searchId, [move])

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

    with open(output_path + f"{start_data.battlePtr}_{dt.isoformat()}.txt", "a") as f:

        stop_watch(st, "Battle started")
        f.write(str(extract_features(obs)) + "\n")

        while True:
            move = agent_blueprint(obs)
            f.write(str(move) + "\n")
            obs = battle_select([move])
            f.write(str(extract_features(obs)) + "\n")
            stop_watch(st, "Move selected")
            if obs is None:
                print("Battle finished.")
                break

        battle_finish()
