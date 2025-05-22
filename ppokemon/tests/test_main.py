import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock, call
import numpy as np

# Assuming ppokemon.main is importable and contains perform_evaluation
# If it's not directly importable due to structure, adjust path or add __init__.py
from ppokemon.main import perform_evaluation, SHOWDOWN_SERVER_CONFIGURATION, BATTLE_FORMAT, LOG_LEVEL 
# We need to import MyRLEnv from its actual location for patching if it's used as a type hint
# or if isinstance checks are done, but for direct patching of 'ppokemon.main.MyRLEnv' it's fine.


class TestMainEvaluation(unittest.IsolatedAsyncioTestCase):

    @patch('builtins.print') # To capture print output
    @patch('ppokemon.main.PPO.load')
    @patch('ppokemon.main.MyRLEnv', new_callable=MagicMock) # Mock the class MyRLEnv
    @patch('ppokemon.main.RandomPlayer') # Mock RandomPlayer constructor
    @patch('ppokemon.main.SimpleHeuristicsPlayer') # Mock SimpleHeuristicsPlayer constructor
    async def test_evaluation_agent_wins_all(self, MockSimpleHeuristicsPlayer, MockRandomPlayer, MockMyRLEnvClass, MockPPOLoad, mock_print):
        # --- Setup Mocks ---
        mock_ppo_model = MagicMock()
        mock_ppo_model.predict.return_value = (MagicMock(), None) # action, state
        MockPPOLoad.return_value = mock_ppo_model

        mock_env_instance = AsyncMock() # MyRLEnv instance is async
        MockMyRLEnvClass.return_value = mock_env_instance

        mock_obs = np.array([0.0]*70, dtype=np.float32) # Dummy observation
        mock_info = {}
        
        # Configure reset: make it an AsyncMock if it's awaited
        mock_env_instance.reset = AsyncMock(return_value=(mock_obs, mock_info))
        
        # Configure step: make it an AsyncMock. Simulate battle ending after 1 step.
        # (obs, reward, terminated, truncated, info)
        mock_env_instance.step = AsyncMock(return_value=(mock_obs, 0.0, True, False, mock_info)) # done=True (terminated)

        # Mock current_battle attributes
        mock_battle_state = MagicMock()
        mock_battle_state.won = True
        mock_battle_state.opponent_fainted_pokemon = 2 # Agent fainted 2 pokemon
        mock_env_instance.current_battle = mock_battle_state
        
        # Mock player constructors to return a simple mock
        MockRandomPlayer.return_value = MagicMock(spec_set=["__class__", "_send_message"]) # spec_set for stricter mocking
        MockSimpleHeuristicsPlayer.return_value = MagicMock(spec_set=["__class__", "_send_message"])

        # --- Call the function ---
        num_eval_battles = 3 # Keep it small for tests
        await perform_evaluation(
            loaded_model_path="dummy_path", # Not used by mock_ppo_model
            train_env_for_loading=None,    # Not used by mock_ppo_model
            battle_format=BATTLE_FORMAT,
            server_configuration=SHOWDOWN_SERVER_CONFIGURATION,
            log_level=LOG_LEVEL,
            num_evaluation_battles=num_eval_battles
        )

        # --- Assertions ---
        # 1. PPO.load called once
        MockPPOLoad.assert_called_once_with("dummy_path", env=None)

        # 2. Opponent instantiation: RandomPlayer (twice, one as placeholder), SimpleHeuristicsPlayer (once)
        self.assertEqual(MockRandomPlayer.call_count, 2)
        MockSimpleHeuristicsPlayer.assert_called_once()
        
        # Check some args for player instantiation (example for one)
        MockRandomPlayer.assert_any_call(
            battle_format=BATTLE_FORMAT,
            server_configuration=SHOWDOWN_SERVER_CONFIGURATION,
            log_level=LOG_LEVEL
        )

        # 3. MyRLEnv instantiation (once per opponent type = 3 times)
        self.assertEqual(MockMyRLEnvClass.call_count, 3)
        # Check that MyRLEnv was called with an opponent instance
        for call_arg in MockMyRLEnvClass.call_args_list:
            self.assertTrue(hasattr(call_arg.kwargs['opponent'], '_send_message'))


        # 4. Battle loop counts (reset and step)
        # reset called once per battle, per opponent. Total 3 opponents * num_eval_battles
        self.assertEqual(mock_env_instance.reset.call_count, 3 * num_eval_battles)
        # step called once per battle (because it returns done=True), per opponent
        self.assertEqual(mock_env_instance.step.call_count, 3 * num_eval_battles)
        
        # 5. predict called once per step
        self.assertEqual(mock_ppo_model.predict.call_count, 3 * num_eval_battles)

        # 6. Check printed output for win rate, avg duration, avg fainted
        # This is more involved, requires inspecting mock_print.call_args_list
        # Example for win rate for one of the opponents (assuming they all behave the same due to mocks)
        # Let's look for the "Results against RandomPlayer:" block
        
        found_random_player_results = False
        for call_args in mock_print.call_args_list:
            output = call_args[0][0] # print is called with a single string argument
            if "Results against RandomPlayer:" in output:
                found_random_player_results = True
            if found_random_player_results and "Win rate: 1.00" in output:
                # Assert on the specific line after "Results against RandomPlayer:"
                self.assertIn(f"Win rate: 1.00 ({num_eval_battles}/{num_eval_battles} wins)", output)
            if found_random_player_results and "Average battle duration:" in output:
                # Each battle is 1 turn because step returns done=True immediately
                self.assertIn("Average battle duration: 1.00 turns", output) 
            if found_random_player_results and "Average opponent Pokemon fainted by PPO agent:" in output:
                self.assertIn("Average opponent Pokemon fainted by PPO agent: 2.00", output)
                break # Found all relevant lines for RandomPlayer
        self.assertTrue(found_random_player_results, "Did not find print output for RandomPlayer results")
        
        # 7. Check eval_env.close() calls
        self.assertEqual(mock_env_instance.close.call_count, 3) # Once per opponent

    @patch('builtins.print')
    @patch('ppokemon.main.PPO.load')
    @patch('ppokemon.main.MyRLEnv', new_callable=MagicMock)
    @patch('ppokemon.main.RandomPlayer')
    @patch('ppokemon.main.SimpleHeuristicsPlayer')
    async def test_evaluation_agent_loses_all(self, MockSimpleHeuristicsPlayer, MockRandomPlayer, MockMyRLEnvClass, MockPPOLoad, mock_print):
        mock_ppo_model = MagicMock()
        mock_ppo_model.predict.return_value = (MagicMock(), None)
        MockPPOLoad.return_value = mock_ppo_model

        mock_env_instance = AsyncMock()
        MockMyRLEnvClass.return_value = mock_env_instance

        mock_obs = np.array([0.0]*70, dtype=np.float32)
        mock_info = {}
        mock_env_instance.reset = AsyncMock(return_value=(mock_obs, mock_info))
        mock_env_instance.step = AsyncMock(return_value=(mock_obs, 0.0, True, False, mock_info))

        mock_battle_state = MagicMock()
        mock_battle_state.won = False # Agent loses
        mock_battle_state.opponent_fainted_pokemon = 0 # Agent fainted 0
        mock_env_instance.current_battle = mock_battle_state
        
        MockRandomPlayer.return_value = MagicMock(spec_set=["__class__", "_send_message"])
        MockSimpleHeuristicsPlayer.return_value = MagicMock(spec_set=["__class__", "_send_message"])

        num_eval_battles = 2
        await perform_evaluation(
            loaded_model_path="dummy_path", train_env_for_loading=None,
            battle_format=BATTLE_FORMAT, server_configuration=SHOWDOWN_SERVER_CONFIGURATION,
            log_level=LOG_LEVEL, num_evaluation_battles=num_eval_battles
        )

        found_results = False
        for call_args in mock_print.call_args_list:
            output = call_args[0][0]
            if "Results against RandomPlayer:" in output: # Check one opponent type
                found_results = True
            if found_results and "Win rate: 0.00" in output:
                self.assertIn(f"Win rate: 0.00 (0/{num_eval_battles} wins)", output)
                break
        self.assertTrue(found_results, "Did not find RandomPlayer results or win rate 0.00 not asserted.")

    @patch('builtins.print')
    @patch('ppokemon.main.PPO.load')
    @patch('ppokemon.main.MyRLEnv', new_callable=MagicMock)
    @patch('ppokemon.main.RandomPlayer')
    @patch('ppokemon.main.SimpleHeuristicsPlayer')
    async def test_evaluation_mixed_results(self, MockSimpleHeuristicsPlayer, MockRandomPlayer, MockMyRLEnvClass, MockPPOLoad, mock_print):
        mock_ppo_model = MagicMock()
        mock_ppo_model.predict.return_value = (MagicMock(), None)
        MockPPOLoad.return_value = mock_ppo_model

        mock_env_instance = AsyncMock()
        MockMyRLEnvClass.return_value = mock_env_instance

        mock_obs = np.array([0.0]*70, dtype=np.float32)
        mock_info = {}
        mock_env_instance.reset = AsyncMock(return_value=(mock_obs, mock_info))

        # Simulate 3 battles: Win (2 turns, 3 fainted), Loss (3 turns, 1 fainted), Win (1 turn, 2 fainted)
        # This setup will apply to one opponent type.
        battle_outcomes = [
            {'won': True, 'fainted': 3, 'steps_to_done': 2}, 
            {'won': False, 'fainted': 1, 'steps_to_done': 3},
            {'won': True, 'fainted': 2, 'steps_to_done': 1}
        ]
        num_eval_battles = len(battle_outcomes)

        # We need step to behave differently across calls for a single opponent
        # And current_battle to also change. This is tricky if we mock MyRLEnv instance once per opponent.
        # Let's simplify: assume MyRLEnv is re-mocked per opponent, or its state is reset.
        # For this test, we'll focus on the logic for *one* opponent and how it averages.
        # The current mock setup reuses mock_env_instance for all opponents.
        # To test varying behavior per battle for one opponent:
        
        step_call_count = 0
        def varying_step_mock(*args, **kwargs):
            nonlocal step_call_count
            current_outcome_index = 0
            # Determine which battle we are in based on reset calls or step calls.
            # This needs careful thought. Let's assume step_mock is for one battle.
            # The problem is, step is called multiple times *within* a battle.
            # We need to control when done becomes True.
            
            # Simpler: mock_env_instance.step needs to count its own calls *per battle*
            # This is easier if we control the number of steps directly.
            
            # Let's make current_battle change based on reset calls.
            # reset_count = mock_env_instance.reset.call_count -1 # 0-indexed
            # current_outcome = battle_outcomes[reset_count % num_eval_battles]
            
            # This still doesn't vary step counts.
            # We need to mock the step function to return done=True after X calls.
            
            # This is getting complicated. Let's simplify the mixed test to focus on the *reporting*
            # given that individual battles have outcomes.
            # We will make current_battle change its attributes upon each reset.
            
            # The test `test_evaluation_agent_wins_all` already tests multiple battles with fixed outcomes.
            # The core logic is `avg = sum() / len()`.
            # Let's verify this for one opponent, with varying battle data.

            # Setup mock_env_instance.current_battle to change on each reset.
            # This requires a side_effect on reset or on current_battle access.
            # Let's make current_battle a property that cycles through outcomes.
            
            # This is still too complex for a single test.
            # The most important part is that if current_battle has values, they are averaged.

            # We will simulate this by having reset configure current_battle for the *next* battle.
            # And step will always make the battle last 1 turn.
            pass # Placeholder for complex step logic if needed.
        
        # For mixed results, we'll make `current_battle` attributes change per call to `reset`.
        # This means `mock_env_instance.current_battle` needs to be dynamic or `reset` needs to set it.
        
        battle_data_iterator = iter(battle_outcomes)
        
        original_reset_side_effect = mock_env_instance.reset.side_effect # Should be None initially if not set
        
        async def side_effect_reset(*args, **kwargs):
            nonlocal battle_data_iterator
            try:
                current_data = next(battle_data_iterator)
                mock_env_instance.current_battle.won = current_data['won']
                # mock_env_instance.current_battle.turn is not how duration is tracked, it's by steps.
                mock_env_instance.current_battle.opponent_fainted_pokemon = current_data['fainted']
                
                # Make step run for current_data['steps_to_done']
                step_responses = [ (mock_obs, 0.0, False, False, mock_info) ] * (current_data['steps_to_done'] -1)
                step_responses.append( (mock_obs, 0.0, True, False, mock_info) ) # Last step, done = True
                mock_env_instance.step = AsyncMock(side_effect=step_responses)

            except StopIteration: # Not enough battle_data for all battles, use defaults
                mock_env_instance.current_battle.won = False
                mock_env_instance.current_battle.opponent_fainted_pokemon = 0
                mock_env_instance.step = AsyncMock(return_value=(mock_obs, 0.0, True, False, mock_info)) # 1 step
            
            if callable(original_reset_side_effect): # Should not be the case here
                 return await original_reset_side_effect(*args, **kwargs)
            return mock_obs, mock_info

        mock_env_instance.reset.side_effect = side_effect_reset
        mock_env_instance.current_battle = MagicMock() # Initial mock_battle
                
        MockRandomPlayer.return_value = MagicMock(spec_set=["__class__", "_send_message"])
        MockSimpleHeuristicsPlayer.return_value = MagicMock(spec_set=["__class__", "_send_message"])

        # We are testing one opponent type (e.g. RandomPlayer) with these mixed results.
        # The `perform_evaluation` will loop 3 times (for 3 opponents).
        # Our side_effect_reset will run out of `battle_outcomes` for the 2nd and 3rd opponent.
        # This is fine, it will default. We care about the first one.
        
        await perform_evaluation(
            loaded_model_path="dummy_path", train_env_for_loading=None,
            battle_format=BATTLE_FORMAT, server_configuration=SHOWDOWN_SERVER_CONFIGURATION,
            log_level=LOG_LEVEL, num_evaluation_battles=num_eval_battles # Should be 3 for battle_outcomes
        )

        # Expected for the first opponent (e.g. RandomPlayer):
        # Wins: 2 (True, False, True)
        # Total battles: 3
        # Win rate: 2/3 = 0.666...
        # Durations: [2, 3, 1]. Avg = (2+3+1)/3 = 6/3 = 2.0
        # Fainted: [3, 1, 2]. Avg = (3+1+2)/3 = 6/3 = 2.0

        expected_win_rate_str = f"Win rate: {2/3:.2f} (2/{num_eval_battles} wins)" # 0.67
        expected_duration_str = "Average battle duration: 2.00 turns"
        expected_fainted_str = "Average opponent Pokemon fainted by PPO agent: 2.00"
        
        # Search for these strings in print output for the first opponent type
        # (which is RandomPlayer by default in perform_evaluation)
        
        found_random_player_results = False
        all_metrics_found_for_random = 0 # Count to 3
        
        for call_args in mock_print.call_args_list:
            output = call_args[0][0]
            if "Results against RandomPlayer:" in output:
                found_random_player_results = True
                continue # Results are on next lines
            
            if found_random_player_results:
                if expected_win_rate_str in output:
                    all_metrics_found_for_random +=1
                if expected_duration_str in output:
                    all_metrics_found_for_random +=1
                if expected_fainted_str in output:
                    all_metrics_found_for_random +=1
                if "---" in output or "Evaluating against:" in output or "Closing evaluation environment" in output : # End of this opponent's block
                    break 
        
        self.assertTrue(found_random_player_results, "Did not find print output for RandomPlayer results in mixed test.")
        self.assertEqual(all_metrics_found_for_random, 3, "Not all expected metrics found or correct for RandomPlayer in mixed test.")


if __name__ == '__main__':
    unittest.main()
