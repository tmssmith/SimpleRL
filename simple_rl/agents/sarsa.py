from typing import Hashable
from simple_rl.agents import QLearningAgent


class SARSAAgent(QLearningAgent):
    def update_q_table(self, state: Hashable, action: tuple, reward, next_state: Hashable, terminal: bool):
        if terminal:
            q_next = 0
        else:
            q_next = self.q[next_state][self.policy(next_state)]
        self.q[state][action] += self.alpha * (reward + self.gamma * q_next - self.q[state][action])
