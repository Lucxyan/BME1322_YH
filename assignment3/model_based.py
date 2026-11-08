"""Tabular model estimation followed by synchronous value iteration."""


class ModelBasedAgent:
    """Learn an empirical MDP from episodes and plan in that MDP."""

    def __init__(self, states, actions, gamma: float = 0.9, seed=None):
        self.states = list(states)
        self.actions = list(actions)
        self.gamma = float(gamma)
        self.seed = seed
        self.values = {state: 0.0 for state in self.states}
        self.counts = {}
        self.transition_counts = {}
        self.reward_sums = {}
        self.transitions = {}
        self.rewards = {}
        self._initialize_model()

    def _initialize_model(self):
        uniform = 1.0 / len(self.states)
        for state in self.states:
            for action in self.actions:
                key = (state, action)
                self.counts[key] = 0
                self.transition_counts[key] = {next_state: 0 for next_state in self.states}
                self.reward_sums[key] = 0.0
                self.transitions[key] = {next_state: uniform for next_state in self.states}
                self.rewards[key] = 0.0

    def learn_model(self, episodes: list):
        """Estimate P(s'|s,a) and E[r|s,a] by counts and sample means."""
        self._initialize_model()
        self.values = {state: 0.0 for state in self.states}
        for episode in episodes:
            for state, action, reward, next_state, _done in episode:
                key = (state, action)
                self.counts[key] += 1
                self.transition_counts[key][next_state] += 1
                self.reward_sums[key] += reward

        for key, count in self.counts.items():
            if count:
                self.transitions[key] = {
                    next_state: visits / count
                    for next_state, visits in self.transition_counts[key].items()
                }
                self.rewards[key] = self.reward_sums[key] / count

    def _action_value(self, state, action, values):
        key = (state, action)
        future = sum(
            probability * values[next_state]
            for next_state, probability in self.transitions[key].items()
        )
        return self.rewards[key] + self.gamma * future

    def run_value_iteration(self, theta: float = 1e-4, max_iterations: int = 1000):
        """Apply the assignment's Bellman equation using synchronous sweeps."""
        for _ in range(max_iterations):
            updated = {
                state: max(self._action_value(state, action, self.values)
                           for action in self.actions)
                for state in self.states
            }
            delta = max(abs(updated[state] - self.values[state]) for state in self.states)
            self.values = updated
            if delta < theta:
                break

    def get_transition(self, state, action) -> dict:
        return dict(self.transitions[(state, action)])

    def get_reward(self, state, action) -> float:
        return float(self.rewards[(state, action)])

    def get_value(self, state) -> float:
        return float(self.values[state])

    def get_policy(self, state):
        # max retains the first action when values tie.
        return max(self.actions, key=lambda action: self._action_value(state, action, self.values))
