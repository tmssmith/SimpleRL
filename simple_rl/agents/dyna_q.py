from typing import Hashable
from numpy.random import Generator as RNG
from collections import defaultdict, Counter
from simple_rl.agents import QLearningAgent


class DynaQAgent(QLearningAgent):
    def __init__(
        self,
        rng: RNG,
        actions: list[Hashable],
        gamma: float = 0.9,
        eps: float = 0.15,
        alpha: float = 0.5,
        num_planning_steps: int = 10,
        default_action_value=0,
        alpha_decay: float = 0,
    ) -> None:
        super().__init__(rng, actions, gamma, eps, alpha, default_action_value, alpha_decay)
        self.num_planning_steps = num_planning_steps
        self.model = defaultdict(lambda: defaultdict(lambda: Counter()))
        # {S: {A: Counter({(R0,S0'): count0, (R1,S1'): count1})}}

    def run(self, env, episodes: int) -> None:
        for episode in range(episodes):
            state = env.reset()
            terminal = False
            step = 0
            while not terminal:
                step += 1
                action = self.policy(state)
                next_state, reward, terminal, _ = env.step(action)
                self.update_q_table(state, action, reward, next_state, terminal)
                self.update_model(state, action, next_state, reward, terminal)
                self.planning(self.num_planning_steps)
                state = next_state
                if self.alpha_decay:
                    self.decay_alpha(step)

    def update_model(self, state, action, next_state, reward, terminal):
        """Updates model based on transition experienced in environment"""
        self.model[state][action][(next_state, reward, terminal)] += 1

    def sample_from_model(self, state, action):
        """Returns the sample next state and reward from taking action in state"""
        total = sum(self.model[state][action].values())
        p = [count / total for count in self.model[state][action].values()]
        return self.rng.choice(list(self.model[state][action]), p=p)

    def planning(self, num_steps: int):
        for n in range(num_steps):
            state = self.rng.choice(list(self.model.keys()))  # Choose random previously visited state
            action = self.rng.choice(list(self.model[state].keys()))  # Choose random action from selected state
            ## Model S', R, T from S,A
            next_state, reward, terminal = self.sample_from_model(state, action)
            self.update_q_table(state, action, reward, next_state, terminal)
