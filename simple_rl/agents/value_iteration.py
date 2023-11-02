from typing import Hashable
from simple_rl.agents import BaseAgent
from collections import defaultdict
from numpy.random import Generator as RNG


class ValueIterationAgent(BaseAgent):
    def __init__(
        self, rng: RNG, actions: list[Hashable], gamma: float, theta: float, default_state_value: float = 0.0
    ) -> None:
        self.rng = rng
        self.gamma = gamma
        self.theta = theta
        self.actions = set(actions)
        self.v = defaultdict(lambda: default_state_value)
        self.policy_table = defaultdict(list)

    def policy(self, state: Hashable) -> Hashable:
        return self.rng.choice(self.policy_table[state])

    def run(self, env, steps: int = None) -> None:
        self.value_iteration(env)

    def get_max_q_value(self, state: Hashable) -> float:
        return max(self.q[state][action] for action in self.actions)

    def get_greedy_actions(self, state: Hashable) -> list[Hashable]:
        return self.policy_table[state]

    def value_iteration(self, env):
        """Performs value iteration"""
        assert hasattr(env, "P"), "Environment does not have transition matrix P"
        assert hasattr(env, "num_states"), "Environment does not have attribute num_states"
        while True:
            delta = 0.0
            # Iterate over all states
            for state in range(env.num_states):
                v_prev = self.v[state]
                v_s = [0.0] * len(self.actions)
                for n, action in enumerate(self.actions):
                    v_s[n] = 0
                    transitions = env.P[state][action]
                    for p, next_state, reward, terminal in transitions:
                        v_s[n] += p * (reward + (1 - terminal) * self.gamma * self.v[next_state])
                self.v[state] = max(v_s)
                delta = max(delta, abs(v_prev - self.v[state]))

            if delta < self.theta:
                break
        self.get_policy(env)

    def get_policy(self, env):
        assert hasattr(env, "P"), "Environment does not have transition matrix P"
        assert hasattr(env, "num_states"), "Environment does not have attribute num_states"
        for state in range(env.num_states):
            v_s = [0.0] * len(self.actions)
            for n, action in enumerate(self.actions):
                transitions = env.P[state][action]
                v_s[n] = 0
                for p, next_state, reward, terminal in transitions:
                    v_s[n] += p * (reward + (1 - terminal) * self.gamma * self.v[next_state])

            self.policy_table[state] += [a for a, value in enumerate(v_s) if value == max(v_s)]
