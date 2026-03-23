"""Tests for the Outcome/step() interface refactor."""
import numpy as np
import pytest

from ibrl.outcome import Outcome
from ibrl.environments.base import BaseEnvironment
from ibrl.environments.bandit import BanditEnvironment
from ibrl.environments.base_newcomb_like import BaseNewcombLikeEnvironment
from ibrl.environments.newcomb import NewcombEnvironment
from ibrl.environments.damascus import DeathInDamascusEnvironment
from ibrl.utils import sample_action


class TestOutcome:
    def test_outcome_creation(self):
        o = Outcome(reward=1.5, env_action=0)
        assert o.reward == 1.5
        assert o.env_action == 0

    def test_outcome_default_env_action(self):
        o = Outcome(reward=1.0)
        assert o.env_action is None


class TestBanditStep:
    def test_step_returns_outcome(self):
        env = BanditEnvironment(num_actions=2, seed=42)
        env.reset()
        outcome = env.step(np.array([0.5, 0.5]), 0)
        assert isinstance(outcome, Outcome)
        assert isinstance(outcome.reward, float)
        assert outcome.env_action is None

    def test_step_deterministic_with_seed(self):
        """Same seed should produce same outcomes."""
        env1 = BanditEnvironment(num_actions=2, seed=42)
        env1.reset()
        env2 = BanditEnvironment(num_actions=2, seed=42)
        env2.reset()

        probs = np.array([0.5, 0.5])
        o1 = env1.step(probs, 0)
        o2 = env2.step(probs, 0)
        assert o1.reward == o2.reward


class TestNewcombStep:
    def test_step_returns_env_action(self):
        env = NewcombEnvironment(num_actions=2, seed=42)
        env.reset()
        probs = np.array([1.0, 0.0])  # always one-box
        outcome = env.step(probs, 0)
        assert isinstance(outcome, Outcome)
        assert outcome.env_action is not None
        assert outcome.env_action in [0, 1]

    def test_env_action_matches_prediction(self):
        """With deterministic policy, predictor should always match."""
        env = NewcombEnvironment(num_actions=2, seed=42)
        env.reset()
        probs = np.array([1.0, 0.0])  # deterministic one-box

        for _ in range(20):
            outcome = env.step(probs, 0)
            assert outcome.env_action == 0, \
                "Predictor should always predict action 0 when p(0)=1"
            # Reward should be reward_table[0, 0] = boxB = 1
            assert outcome.reward == 1.0

    def test_reward_matches_table(self):
        """Reward should be reward_table[env_action, action]."""
        env = NewcombEnvironment(num_actions=2, seed=42, boxA=0.1, boxB=1)
        env.reset()
        # Force predictor to predict 0 by using p=[1, 0]
        probs = np.array([1.0, 0.0])
        outcome = env.step(probs, 0)
        assert outcome.env_action == 0
        assert outcome.reward == 1.0  # reward_table[0, 0] = boxB

    def test_damascus_step(self):
        env = DeathInDamascusEnvironment(num_actions=2, seed=42)
        env.reset()
        probs = np.array([1.0, 0.0])
        outcome = env.step(probs, 0)
        # Predictor predicts 0, agent plays 0 -> death (reward 0)
        assert outcome.env_action == 0
        assert outcome.reward == 0.0
