"""Offline tabular Q-learning, with an optional online action selector."""

import random


class QLearningAgent:
    def __init__(self, states, actions, gamma: float = 0.9, alpha: float = 0.5,
                 epsilon: float = 0.3, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05, seed=None):
        self.states = list(states)
        self.actions = list(actions)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.rng = random.Random(seed)
        self.q = {(state, action): 0.0 for state in self.states for action in self.actions}

    def learn(self, episodes: list, mode: str = "fixed"):
        """Replay the supplied experiences in order; mode is protocol-only."""
        for episode in episodes:
            for state, action, reward, next_state, done in episode:
                self.update(state, action, reward, next_state, done)

    def update(self, state, action, reward, next_state, done: bool):
        future = 0.0 if done else max(self.q[(next_state, a)] for a in self.actions)
        target = reward + self.gamma * future
        key = (state, action)
        self.q[key] += self.alpha * (target - self.q[key])

    def get_q_value(self, state, action) -> float:
        return float(self.q[(state, action)])

    def get_policy(self, state):
        # max retains the first action when values tie.
        return max(self.actions, key=lambda action: self.q[(state, action)])

    def select_action(self, state):
        """Epsilon-greedy selection for the report's separate online experiment."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.get_policy(state)
