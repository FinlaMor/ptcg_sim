import csv
import json
from pathlib import Path

INPUT_DIR = Path("games/scrapped")
OUTPUT_FILE = Path("training_data.csv")

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


def observation_to_row(file_name, step_index, entry_index, entry, rewards):
    obs = entry["observation"]
    current = obs.get("current")
    if current is None:
        return None

    your_index = current.get("yourIndex")
    if your_index is None:
        return None

    if not isinstance(rewards, list) or your_index >= len(rewards):
        return None

    label = rewards[your_index]
    if label not in (-1, 0, 1):
        return None

    row = {
        "file": file_name,
        "step_index": step_index,
        "entry_index": entry_index,
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
        "label": 1 if label == 1 else 0,
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


def main():
    rows = []
    input_files = sorted(INPUT_DIR.glob("*.json"))

    for path in input_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rewards = data.get("rewards", [])
        for step_index, step in enumerate(data.get("steps", [])):
            for entry_index, entry in enumerate(step):
                if entry.get("status") != "ACTIVE":
                    continue
                row = observation_to_row(path.name, step_index, entry_index, entry, rewards)
                if row is not None:
                    rows.append(row)

    if not rows:
        raise ValueError("No training rows were generated.")

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
