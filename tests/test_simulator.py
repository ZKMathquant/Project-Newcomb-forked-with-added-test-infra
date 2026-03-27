import numpy as np
import pytest
from ibrl.simulators import simulate
from ibrl.agents import QLearningAgent
from ibrl.environments import BanditEnvironment


class TestSimulator:
    def test_simulate_runs_without_error(self, num_actions, seed):
        env = BanditEnvironment(num_actions=num_actions, seed=seed)
        agent = QLearningAgent(num_actions=num_actions, seed=seed + 1)
        options = {"num_steps": 10, "num_runs": 2}
        results = simulate(env, agent, options)
        assert "average_reward" in results
        assert "optimal_reward" in results

    def test_simulate_output_shapes(self, num_actions, seed):
        env = BanditEnvironment(num_actions=num_actions, seed=seed)
        agent = QLearningAgent(num_actions=num_actions, seed=seed + 1)
        num_steps, num_runs = 10, 2
        options = {"num_steps": num_steps, "num_runs": num_runs}
        results = simulate(env, agent, options)
        
        assert results["average_reward"].shape == (2, num_steps)
        assert results["probabilities"].shape == (num_runs, num_steps, num_actions)
        assert results["actions"].shape == (num_runs, num_steps)
        assert results["rewards"].shape == (num_runs, num_steps)

    def test_simulate_rewards_valid(self, num_actions, seed):
        env = BanditEnvironment(num_actions=num_actions, seed=seed)
        agent = QLearningAgent(num_actions=num_actions, seed=seed + 1)
        options = {"num_steps": 10, "num_runs": 2}
        results = simulate(env, agent, options)
        
        assert np.all(np.isfinite(results["rewards"]))
        assert np.all(np.isfinite(results["average_reward"]))
