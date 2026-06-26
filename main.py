import lightgbm as lgb
import numpy as np

bst = lgb.Booster(model_file='pok_model.txt')

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

def agent_blueprint(observation, configuration):

    current_state = observation.state
    legal_moves = observation.legal_moves

    best_move = None
    highest_score = -float('inf')

    for move_index in legal_moves:
        hypothetical_state = cabt_engine.step(current_state, move_index)

        board_features = extract_features(hypothetical_state)

        score = bst.predict([board_features])[0]

        if score > highest_score:
            highest_score = score
            best_move = move_index
            
    return best_move