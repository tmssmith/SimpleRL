from typing import Hashable
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Base interface for tabular RL agents.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self, env, steps: int) -> None:
        """Runs agent in environment for a given number of steps.

        Args:
            env: environment to run in.
            steps (int): number of steps to run in environment.
        """
        pass

    @abstractmethod
    def policy(self, state: Hashable) -> Hashable:
        """
        Returns the action specified by the agent's policy in a given state.

        Args:
            state (Hashable): state in which the agent is acting.

        Returns:
            Hashable: the action chosen by the agent's policy in a given state.
        """
        pass

    @abstractmethod
    def get_greedy_actions(self, state: Hashable) -> list[Hashable]:
        """
        Returns a list of actions that could be taken under the agent's current greedy policy.

        Args:
            state (Hashable): state in which the agent is acting.

        Returns:
            list[Hashable]: list of actions that could be selected under the agent's greedy policy.
        """
        pass
