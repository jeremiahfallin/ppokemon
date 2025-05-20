import asyncio
import logging
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.player import Player
from poke_env.player.singles_env import SinglesEnv
from typing import Tuple, Dict, Any, Optional

SHOWDOWN_SERVER_CONFIGURATION = ServerConfiguration(
    "ws://showdown:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
BATTLE_FORMAT = "gen9randombattle"
LOG_LEVEL = logging.ERROR


class MyRLEnv(SinglesEnv, gymnasium.Env):
    def __init__(self, battle_format: str, **kwargs):
        super().__init__(
            battle_format=battle_format,
            start_challenging=True,
            **kwargs,
        )

        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(70,), dtype=np.float32
        )

        expected_action_size = 18
        if hasattr(self, "_ACTION_SPACE") and isinstance(self._ACTION_SPACE, list):
            expected_action_size = len(self._ACTION_SPACE)
        elif hasattr(self, "action_space_converter") and isinstance(
            self.action_space_converter, list
        ):
            expected_action_size = len(self.action_space_converter)

        self.action_space = gymnasium.spaces.Discrete(expected_action_size)

    def embed_battle(
        self, battle: AbstractBattle
    ) -> np.ndarray:  # Or your specific ObsType
        """
        Convert the battle state to a numerical vector.
        This is where you extract all relevant features.
        """
        # ----- YOUR FEATURE ENGINEERING GOES HERE -----
        # Example features (VERY simplified - you need many more):
        features = []

        # Active Pokémon features (yours)
        if battle.active_pokemon:
            features.append(battle.active_pokemon.current_hp_fraction)
            features.append(1.0 if battle.active_pokemon.status is not None else 0.0)
            # Add stats, type, moves (presence, PP, type, power), boosts etc.
            # One-hot encode types, statuses, move types. Normalize stats.
        else:  # Should not happen in a normal battle unless it's over
            features.extend([0.0] * 2)  # Placeholder for 2 features

        # Opponent's active Pokémon features
        if battle.opponent_active_pokemon:
            features.append(battle.opponent_active_pokemon.current_hp_fraction)
            features.append(
                1.0 if battle.opponent_active_pokemon.status is not None else 0.0
            )
            # Add known stats, types, moves, boosts etc.
        else:
            features.extend([0.0] * 2)  # Placeholder

        # Team status (yours and opponent's)
        # e.g., number of fainted pokemon, status of benched pokemon

        # Field conditions
        # e.g., weather, terrain, hazards (one-hot encode or binary flags)

        # Ensure the feature vector always has the same length
        # Pad with zeros if necessary, or have fixed slots for all possible info.
        # For this example, let's assume we've designed it to be 70 features.
        # This part requires careful design!

        # --- This is a placeholder for a real feature vector ---
        # Make sure the number of features matches self.observation_space.shape[0]
        current_feature_count = len(features)
        if current_feature_count < 70:
            features.extend([0.0] * (70 - current_feature_count))
        elif current_feature_count > 70:
            features = features[:70]

        return np.array(features, dtype=np.float32)

    def calc_reward(self, battle: AbstractBattle) -> float:

        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        print("DEBUG: MyRLEnv.reset() called")
        obs, info = super().reset(seed=seed, options=options)
        print(f"DEBUG: MyRLEnv.reset() - obs from super: {type(obs)}")

        if not isinstance(obs, np.ndarray):
            if isinstance(obs, AbstractBattle):
                print("DEBUG: MyRLEnv.reset() - obs was AbstractBattle, re-embedding.")
                obs = self.embed_battle(obs)
                print(
                    f"DEBUG: MyRLEnv.reset() - obs type after re-embedding: {type(obs)}"
                )
            else:
                print(
                    f"DEBUG: MyRLEnv.reset() - WARNING: obs is {type(obs)}, not np.ndarray or AbstractBattle. Returning zero array."
                )
                obs = np.zeros(
                    self.observation_space.shape, dtype=self.observation_space.dtype
                )

        if not isinstance(info, dict):
            print(
                f"DEBUG: MyRLEnv.reset() - WARNING: info is {type(info)}, not dict. Resetting to empty dict."
            )
            info = {}

        print(
            f"DEBUG: MyRLEnv.reset() - final obs type: {type(obs)}, info type: {type(info)}"
        )
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        print(f"DEBUG: MyRLEnv.step({action}) called")
        obs, reward, terminated, truncated, info = super().step(action)
        print(f"DEBUG: MyRLEnv.step() - obs from super: {type(obs)}")

        if not isinstance(obs, np.ndarray):
            if isinstance(obs, AbstractBattle):
                print("DEBUG: MyRLEnv.step() - obs was AbstractBattle, re-embedding.")
                obs = self.embed_battle(obs)
                print(
                    f"DEBUG: MyRLEnv.step() - obs type after re-embedding: {type(obs)}"
                )
            else:
                print(
                    f"DEBUG: MyRLEnv.step() - WARNING: obs is {type(obs)}, not np.ndarray or AbstractBattle. Returning zero array."
                )
                obs = np.zeros(
                    self.observation_space.shape, dtype=self.observation_space.dtype
                )

        # Type checking for SB3 compatibility, with conversions/defaults
        if not isinstance(reward, float):
            print(
                f"DEBUG: MyRLEnv.step() - WARNING: reward is {type(reward)}, not float. Converting."
            )
            try:
                reward = float(reward)
            except (ValueError, TypeError):
                reward = 0.0
        if not isinstance(terminated, bool):
            print(
                f"DEBUG: MyRLEnv.step() - WARNING: terminated is {type(terminated)}, not bool. Converting."
            )
            terminated = bool(terminated)
        if not isinstance(truncated, bool):
            print(
                f"DEBUG: MyRLEnv.step() - WARNING: truncated is {type(truncated)}, not bool. Converting."
            )
            truncated = bool(truncated)
        if not isinstance(info, dict):
            print(
                f"DEBUG: MyRLEnv.step() - WARNING: info is {type(info)}, not dict. Resetting to empty dict."
            )
            info = {}

        print(
            f"DEBUG: MyRLEnv.step() - final obs type: {type(obs)}, reward: {reward}, term: {terminated}, trunc: {truncated}, info type: {type(info)}"
        )
        return obs, float(reward), terminated, truncated, info


if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)  # Set a global log level
    # Reduce websockets library verbosity (it's very chatty on INFO)
    logging.getLogger("websockets.client").setLevel(LOG_LEVEL)
    print(
        "ppokemon.py executed directly (should not happen in normal Docker execution)"
    )
    # Example: to test MyRLEnv directly (requires an event loop)
    # async def test_env():
    #     from poke_env.player import RandomPlayer
    #     opponent = RandomPlayer(battle_format=BATTLE_FORMAT)
    #     env = MyRLEnv(battle_format=BATTLE_FORMAT, opponent=opponent, server_configuration=SHOWDOWN_SERVER_CONFIGURATION)
    #     check_env(env)
    #     env.close()
    # asyncio.run(test_env())
