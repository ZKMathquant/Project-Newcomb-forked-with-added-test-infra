"""Tests for the UCB1 agent."""
import numpy as np
import pytest

from ibrl.agents.ucb import UCBAgent
from ibrl.outcome import Outcome
from ibrl.utils.construction import construct_agent, construct_environment
from ibrl.simulators.simulator import simulate


class TestUCBConstruction:
    def test_construct_via_string(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        agent = construct_agent("ucb", options)
        assert isinstance(agent, UCBAgent)

    def test_construct_with_exploration(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        agent = construct_agent("ucb:exploration=1.0", options)
        assert agent.exploration == 1.0

    def test_reset(self):
        agent = UCBAgent(num_actions=3, seed=42)
        agent.reset()
        np.testing.assert_array_equal(agent.q, [0, 0, 0])
        np.testing.assert_array_equal(agent.counts, [0, 0, 0])


class TestUCBPolicy:
    def test_initial_policy_is_one_hot(self):
        """UCB always returns a one-hot distribution."""
        agent = UCBAgent(num_actions=3, seed=42)
        agent.reset()
        probs = agent.get_probabilities()
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-10
        assert abs(probs.max() - 1.0) < 1e-10  # one-hot

    def test_round_robin_initialization(self):
        """UCB pulls each arm once before using UCB formula."""
        agent = UCBAgent(num_actions=3, seed=42)
        agent.reset()

        # First three pulls should be arms 0, 1, 2
        for i in range(3):
            probs = agent.get_probabilities()
            assert probs[i] == 1.0, f"Step {i}: expected arm {i}"
            agent.update(probs, i, Outcome(reward=float(i)))

    def test_favors_high_reward_arm(self):
        """After enough pulls, UCB should prefer the arm with higher reward."""
        agent = UCBAgent(num_actions=2, seed=42, exploration=2.0)
        agent.reset()

        # Pull arm 0 (low reward) and arm 1 (high reward)
        p = agent.get_probabilities()
        agent.update(p, 0, Outcome(reward=0.1))
        p = agent.get_probabilities()
        agent.update(p, 1, Outcome(reward=0.9))

        # Need enough pulls for the mean difference (0.8) to overcome
        # the exploration bonus for arm 0. With c=2:
        # UCB(0) = 0.1 + 2*sqrt(ln(t)/1), UCB(1) = 0.9 + 2*sqrt(ln(t)/n1)
        # Arm 0 gets explored when its UCB is higher, eventually arm 1 wins
        for _ in range(200):
            p = agent.get_probabilities()
            chosen = np.argmax(p)
            reward = 0.1 if chosen == 0 else 0.9
            agent.update(p, chosen, Outcome(reward=reward))

        # After 200 steps, arm 1 should be preferred (much higher mean)
        probs = agent.get_probabilities()
        assert probs[1] == 1.0, \
            f"UCB should prefer high-reward arm, got probs={probs}"


class TestUCBOnBandit:
    def test_sublinear_regret(self):
        """UCB should achieve sublinear regret on standard bandits."""
        options = {"num_actions": 2, "num_steps": 1000, "num_runs": 20, "seed": 42, "verbose": 0}
        env = construct_environment("bandit", dict(options))
        agent = construct_agent("ucb", dict(options))
        results = simulate(env, agent, options)

        cumulative = results["average_reward"][0].cumsum()
        optimal_cumulative = results["optimal_reward"] * np.arange(1, 1001)
        regret = optimal_cumulative - cumulative

        # Regret should be sublinear: regret/N should decrease
        assert regret[999] / 1000 < 0.1, \
            f"Regret rate too high: {regret[999]/1000:.3f}"


class TestUCBOnDamascus:
    def test_fails_on_damascus(self):
        """UCB should fail on Damascus (gets ~0 reward, optimal is 5)."""
        options = {"num_actions": 2, "num_steps": 200, "num_runs": 10, "seed": 42, "verbose": 0}
        env = construct_environment("damascus", dict(options))
        agent = construct_agent("ucb", dict(options))
        results = simulate(env, agent, options)

        avg_reward = results["average_reward"][0][-50:].mean()
        # UCB plays pure strategy, predictor matches -> always death (reward 0)
        assert avg_reward < 2.0, \
            f"UCB should fail on Damascus but got avg_reward={avg_reward:.2f}"
