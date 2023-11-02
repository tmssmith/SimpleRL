from typing import Hashable
from simple_rl.agents import BaseAgent
from collections import defaultdict
from numpy.random import Generator as RNG
import matplotlib.animation as animation
import matplotlib.pyplot as plt


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        rng: RNG,
        actions: list[Hashable],
        gamma: float = 0.9,
        eps: float = 0.15,
        alpha: float = 0.5,
        default_action_value=0.0,
        alpha_decay: float = 0.0,
    ) -> None:
        """An agent-centric representation, where the agent is initiated with a set of actions which can be updated.

        Args:
            rng (RNG): random number generator.
            actions (list[Hashable]): list of actions available to the agent.
            gamma (float, optional): discount factor. Defaults to 0.9.
            eps (float, optional): exploration rate of e-greedy policy. Defaults to 0.15.
            alpha (float, optional): learning rate. Defaults to 0.5.
            alpha_decay (float, optional): alpha decay parameter. Defaults to 0.0 (no decay).
        """
        self.gamma = gamma
        self.eps = eps
        self.alpha = self.alpha_init = alpha
        self.alpha_decay = alpha_decay
        self.rng = rng
        self.actions = set(actions)
        self.q = defaultdict(
            lambda: defaultdict(lambda: default_action_value)
        )  # Q(s,w), initalised to default action value.

    def run(self, env, steps: int) -> None:
        state = env.reset()
        terminal = False
        for step in range(steps):
            action = self.policy(state)
            next_state, reward, terminal, _ = env.step(action)
            self.update_q_table(state, action, reward, next_state, terminal)
            if self.alpha_decay:
                self.decay_alpha(step)
            if terminal:
                state = env.reset()
            else:
                state = next_state

    def policy(self, state: Hashable) -> Hashable:
        if self.rng.random() <= self.eps:
            return self.rng.choice(sorted(self.actions))
        else:
            return self.rng.choice(self.get_greedy_actions(state))

    def get_max_q_value(self, state: Hashable) -> float:
        return max(self.q[state][action] for action in self.actions)

    def get_greedy_actions(self, state: Hashable) -> list[Hashable]:
        # Find max q-values in state: Q(s,.)
        max_q_value = self.get_max_q_value(state)
        # Return actions with max value.
        return [action for action, value in self.q[state].items() if value == max_q_value]

    def update_q_table(self, state: Hashable, action: tuple, reward: float, next_state: Hashable, terminal: bool):
        if terminal:
            q_next = 0
        else:
            q_next = self.get_max_q_value(next_state)
        self.q[state][action] += self.alpha * (reward + self.gamma * q_next - self.q[state][action])

    def decay_alpha(self, step):
        self.alpha = 1 / (1 + self.alpha_decay * step) * self.alpha_init
