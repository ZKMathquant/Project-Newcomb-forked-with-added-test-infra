"""Infradistribution class — direct port from coin-learning.ipynb."""
from .helpers import glue


def _reward_zero(history):
    return 0

def _reward_one(history):
    return 1


class Infradistribution:
    """An infradistribution, represented by its extremal minimal points."""

    def __init__(self, measures):
        self.measures = measures

    def expected_value(self, reward_function: callable) -> float:
        """Expected value = min over all a-measures."""
        return min(m.expected_value(reward_function) for m in self.measures)

    def probability(self, reward_function: callable, event) -> float:
        """'Probability' of event under reward function (not a classical probability)."""
        return (
            self.expected_value(glue(_reward_one, event, reward_function))
            - self.expected_value(glue(_reward_zero, event, reward_function))
        )

    def update(self, reward_function: callable, event) -> None:
        """Update all a-measures upon observing event. Definition 11 from Basic Inframeasure Theory."""
        expect0 = self.expected_value(glue(_reward_zero, event, reward_function))
        expect1 = self.expected_value(glue(_reward_one, event, reward_function))
        prob = expect1 - expect0

        for measure in self.measures:
            expect_m = measure.expected_value(glue(_reward_zero, event, reward_function))
            measure.chop(event)
            measure.offset += expect_m - expect0
            measure /= prob

    def __repr__(self) -> str:
        return repr(self.measures)
