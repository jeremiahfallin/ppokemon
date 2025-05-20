# In ./ppokemon/main.py
import asyncio
import logging
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration
from ppokemon import MyRLEnv

# --- Configuration ---
SHOWDOWN_SERVER_CONFIGURATION = ServerConfiguration(
    "ws://showdown:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
BATTLE_FORMAT = "gen9randombattle"
LOG_LEVEL = logging.INFO
MODEL_NAME = "ppo_pokemon_agent"
TRAIN_TIMESTEPS = 100000


async def main():
    start_time = time.time()

    opponent_for_training = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        server_configuration=SHOWDOWN_SERVER_CONFIGURATION,
        log_level=LOG_LEVEL,
    )

    # Create the training environment
    train_env = MyRLEnv(
        battle_format=BATTLE_FORMAT,
        server_configuration=SHOWDOWN_SERVER_CONFIGURATION,
        log_level=LOG_LEVEL,
    )

    print("Checking environment...")
    check_env(train_env)
    print("Environment check (skipped or passed).")

    print(f"Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log="./pokemon_tensorboard/",
        # You can add learning_rate, n_steps, batch_size, etc.
        # For GPU usage with PyTorch (default for SB3):
        # device="cuda" if torch.cuda.is_available() else "cpu"
        # (ensure torch is imported and CUDA is available in Docker)
    )

    # Train the agent
    print(f"Starting training for {TRAIN_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    print("Training finished.")

    # Save the trained model
    model.save(MODEL_NAME)
    print(f"Model saved as {MODEL_NAME}.zip")

    # Clean up the environment
    # This will also stop the battles and disconnect the players.
    print("Closing training environment...")
    train_env.close()  # Important!

    end_time = time.time()
    print(f"Total script time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=LOG_LEVEL)
    # Reduce verbosity of some libraries if needed
    logging.getLogger("websockets.client").setLevel(logging.WARNING)
    # For PyTorch + SB3, you might want to control its specific loggers if too verbose

    # The main() function is now async due to PokeEnv
    asyncio.run(main())  # Use asyncio.run for Python 3.7+
